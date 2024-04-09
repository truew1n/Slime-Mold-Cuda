#include "cuda_runtime.h"
#include <curand_kernel.h>
#include "device_launch_parameters.h"

#include <stdio.h>
#include <windows.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <chrono>
#include <d3d11.h>


#define ERROR_REGISTER_FAILED -2

typedef struct FPoint2D {
    float x;
    float y;
} FPoint2D;

typedef struct Agent {
    FPoint2D Position;
    float Rotation;
    // float PheromoneStrength;
} Agent;

typedef struct UDimension2D {
    uint32_t Width;
    uint32_t Height;
} UDimension2D;

typedef enum CollisionType {
    NONE,
    LEFT,
    RIGHT,
    UP,
    DOWN
} CollisionType;

uint32_t CalcAOB(uint32_t Value, uint32_t AOT)
{
    return (Value / AOT) + ((Value % AOT) > 0);
}

#define __multi__ __host__ __device__

__device__ float Mapf(float Number, float LeftRange, float RightRange, float LeftBound, float RightBound)
{
    return (Number - LeftRange) / (RightRange - LeftRange) * (RightBound - LeftBound) + LeftBound;
}

__device__ float LinInter(float x, float y, float s)
{
    return x + s * (y - x);
}

__device__ float ToBipolar(float Unipolar)
{
    return (Unipolar - 0.5f) * 2.0f;
}

__device__ float ToDiscreteBipolar(float Unipolar)
{
    return (Unipolar > 0.5f ? 1.0f : -1.0f);
}

__device__ uint32_t DecodeRGB(uint8_t R, uint8_t G, uint8_t B)
{
    return (R << 16) + (G << 8) + B;
}

__device__ float AddAngles(float Angle0, float Angle1) {
    return fmodf(((Angle0 + Angle1) + M_PI), (2*M_PI)) - M_PI;
}

__device__ float AgentVelocity = 1.0f;
__device__ float AgentTurnSpeed = 0.0174532925f * 45.0f;
__device__ float AgentSensorLength = 4.0f;
__device__ float AgentSensorAngle = 0.0174532925f * 45.0f;  
#define AgentSensorSize 1
__device__ float DecayRate = 0.7f;
#define DiffusionSize 1
__device__ float DiffusionRate = 0.2f;

__device__ float PheromoneFunction(float X)
{
    if(X > 0.5f) return X * X + 0.5f;
    else return 0.2f * X + 0.4f;
}

__global__ void InitAgents(Agent *Agents, UDimension2D Dimension, uint32_t AgentCount)
{
    uint32_t LinearIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if(LinearIndex >= AgentCount) return;

    curandState state;
    curand_init(blockIdx.x * blockDim.x + threadIdx.x, 0, 0, &state);

    // float RandomRadius = curand_uniform(&state) * 150 + 150;
    float RandomRadius = curand_uniform(&state) * 300;
    float RandomAngle = ToBipolar(curand_uniform(&state)) * M_PI;



    Agents[LinearIndex].Position = {
        (Dimension.Width / 2.0f) + (RandomRadius * cos(RandomAngle)),
        (Dimension.Height / 2.0f) + (RandomRadius * -sin(RandomAngle))
    };

    Agents[LinearIndex].Rotation = AddAngles(RandomAngle, M_PI);
    // Agents[LinearIndex].PheromoneStrength = PheromoneFunction(curand_uniform(&state));
}


__device__ void ResolveBoundCollisions(Agent *Agent, UDimension2D Dimension)
{
    FPoint2D AgentPosition = Agent->Position;
    float AgentRotation = Agent->Rotation;

    CollisionType XCollision = CollisionType::NONE;
    CollisionType YCollision = CollisionType::NONE;


    if(AgentPosition.x < 0.0f) {
        XCollision = CollisionType::LEFT;
    } else if(AgentPosition.x >= Dimension.Width) {
        XCollision = CollisionType::RIGHT;
    }

    if(AgentPosition.y < 0.0f) {
        YCollision = CollisionType::UP;
    } else if(AgentPosition.y >= Dimension.Height) {
        YCollision = CollisionType::DOWN;
    }

    if(XCollision == CollisionType::NONE && YCollision == CollisionType::NONE) return;

    FPoint2D NewAgentPosition = AgentPosition;
    float NewAgentRotation = 0.0f;

    switch(XCollision) {
        case CollisionType::LEFT: {
            NewAgentPosition.x = 0.0f;
            NewAgentRotation = M_PI - AgentRotation;
            break;
        }
        case CollisionType::RIGHT: {
            NewAgentPosition.x = Dimension.Width - 1;
            NewAgentRotation = -AgentRotation + M_PI;
            break;
        }
    }

    switch(YCollision) {
        case CollisionType::UP: {
            NewAgentPosition.y = 0.0f;
            NewAgentRotation = -AgentRotation;
            break;
        }
        case CollisionType::DOWN: {
            NewAgentPosition.y = Dimension.Height - 1;
            NewAgentRotation = -AgentRotation;
            break;
        }
    }

    Agent->Position = NewAgentPosition;
    Agent->Rotation = NewAgentRotation;
}

__device__ float Sense(FPoint2D AgentPosition, float AgentRotation, float Angle, void *TrailMap, UDimension2D Dimension)
{
    float NewAngle = AddAngles(AgentRotation, Angle);
    FPoint2D SensePosition = {
        AgentPosition.x + (AgentSensorLength * cos(NewAngle)),
        AgentPosition.y + (AgentSensorLength * -sin(NewAngle))
    };

    uint32_t LinearSensePosition = (int)(SensePosition.y) * Dimension.Width + (int)(SensePosition.x);

    uint32_t DimensionSize = Dimension.Width * Dimension.Height;

    float Sum = 0.0f;
    for(int32_t j = -AgentSensorSize; j <= AgentSensorSize; ++j) {
        for(int32_t i = -AgentSensorSize; i <= AgentSensorSize; ++i) {
            int32_t SampleLinearIndex = LinearSensePosition + (j * Dimension.Width) + i;
            if(SampleLinearIndex >= 0 && SampleLinearIndex < DimensionSize) {
                Sum += ((float *) TrailMap)[SampleLinearIndex];
            }
        }
    }

    return Sum;
}

__global__ void UpdateAgents(float DeltaTime, Agent *Agents, uint32_t AgentCount, void *TrailMap, UDimension2D Dimension)
{
    uint32_t LinearIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if(LinearIndex >= AgentCount) return;

    FPoint2D CurrentPosition = Agents[LinearIndex].Position;
    float CurrentRotation = Agents[LinearIndex].Rotation;
    FPoint2D NewPosition = {0};

    curandState state;
    curand_init(LinearIndex, 0, 0, &state);
    
    float ForwardSensorWeight = Sense(CurrentPosition, CurrentRotation, 0, TrailMap, Dimension);
    float LeftSensorWeight = Sense(CurrentPosition, CurrentRotation, AgentSensorAngle, TrailMap, Dimension);
    float RightSensorWeight = Sense(CurrentPosition, CurrentRotation, -AgentSensorAngle, TrailMap, Dimension);
    
    // printf("Forward = %f, Left = %f, Right = %f\n",
    //     ForwardSensorWeight,
    //     LeftSensorWeight,
    //     RightSensorWeight
    // );

    float RandomSteerStrength = curand_uniform(&state);

    if(ForwardSensorWeight > LeftSensorWeight && ForwardSensorWeight > RightSensorWeight) {
        CurrentRotation = CurrentRotation;
    } else if(ForwardSensorWeight < LeftSensorWeight && ForwardSensorWeight < RightSensorWeight) {
        CurrentRotation = AddAngles(
            CurrentRotation,
            ToDiscreteBipolar(RandomSteerStrength) * AgentTurnSpeed
        );
    } else if(RightSensorWeight > LeftSensorWeight) {
        CurrentRotation = AddAngles(
            CurrentRotation,
            -AgentTurnSpeed
        );
    } else if(LeftSensorWeight > RightSensorWeight) {
        CurrentRotation = AddAngles(
            CurrentRotation,
            AgentTurnSpeed 
        );
    }

    NewPosition.x = CurrentPosition.x + (AgentVelocity * cos(CurrentRotation));
    NewPosition.y = CurrentPosition.y + (AgentVelocity * -sin(CurrentRotation));

    Agents[LinearIndex].Position = NewPosition;

    ResolveBoundCollisions(&Agents[LinearIndex], Dimension);
}

__global__ void Render(Agent *Agents, uint32_t AgentCount, void *TrailMap, UDimension2D Dimension)
{
    uint32_t LinearIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if(LinearIndex >= AgentCount) return;

    FPoint2D AgentPosition = Agents[LinearIndex].Position;
    
    uint32_t TrailMapIndex = (int)(AgentPosition.y) * Dimension.Width + (int)(AgentPosition.x);
    ((float *) TrailMap)[TrailMapIndex] = 1.0f;
    // ((float *) TrailMap)[TrailMapIndex] = Agents[LinearIndex].PheromoneStrength;
}

__global__ void ProcessTrailMap(float DeltaTime, void *TrailMap, UDimension2D Dimension)
{
    uint32_t LinearIndex = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t X = LinearIndex % Dimension.Width;
    uint32_t Y = LinearIndex / Dimension.Width;

    if(X >= Dimension.Width || Y >= Dimension.Height) return;

    float OriginalValue = ((float *) TrailMap)[LinearIndex];

    float Sum = 0.0f;
    
    uint32_t DimensionSize = Dimension.Width * Dimension.Height;

    for(int32_t j = -DiffusionSize; j <= DiffusionSize; ++j) {
        for(int32_t i = -DiffusionSize; i <= DiffusionSize; ++i) {
            int32_t SampleIndex = LinearIndex + (j * Dimension.Width) + i;
            if(SampleIndex >= 0 && SampleIndex < DimensionSize) {
                Sum += ((float *) TrailMap)[SampleIndex];
            }
        }
    }
    float Blured = Sum / ((DiffusionSize * 2 + 1) * (DiffusionSize * 2 + 1));

    float Diffused = LinInter(OriginalValue, Blured, DiffusionRate);
    float DecayedAndDiffused = Diffused * DecayRate;

    ((float *) TrailMap)[LinearIndex] = DecayedAndDiffused;
}

__global__ void RenderTrailMap(void *TrailMap, void *Display, UDimension2D Dimension)
{
    uint32_t LinearIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if(LinearIndex >= (Dimension.Width * Dimension.Height)) return;

    uint8_t TrailMapColor = (uint8_t)(((float *) TrailMap)[LinearIndex] * 255);

    ((uint32_t *) Display)[LinearIndex] = DecodeRGB(TrailMapColor, 0, TrailMapColor);
}

LRESULT CALLBACK WinProcedure(HWND HWnd, UINT UMsg, WPARAM WParam, LPARAM LParam);

int main(void)
{
    HINSTANCE WinInstance = GetModuleHandleW(NULL);
    
    WNDCLASSW WinClass = {0};
    WinClass.lpszClassName = L"Slime-Mold-Simulation";
    WinClass.hbrBackground = (HBRUSH) COLOR_WINDOW;
    WinClass.hCursor = LoadCursor(NULL, IDC_ARROW);
    WinClass.hInstance = WinInstance;
    WinClass.lpfnWndProc = WinProcedure;

    if(!RegisterClassW(&WinClass)) return ERROR_REGISTER_FAILED;

    uint32_t Width = 800;
    uint32_t Height = 600;

    RECT WindowRect = { 0 };
    WindowRect.right = Width;
    WindowRect.bottom = Height;
    WindowRect.left = 0;
    WindowRect.top = 0;

    AdjustWindowRect(&WindowRect, WS_OVERLAPPEDWINDOW | WS_VISIBLE, 0);
    HWND Window = CreateWindowW(
        WinClass.lpszClassName,
        L"Slime Mold Simulation",
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT,
        WindowRect.right - WindowRect.left,
        WindowRect.bottom - WindowRect.top,
        NULL, NULL,
        NULL, NULL
    );

    GetWindowRect(Window, &WindowRect);

    uint32_t BitmapWidth = Width;
    uint32_t BitmapHeight = Height;

    uint32_t BytesPerPixel = 4;

    uint32_t BitmapTotalSize = BitmapWidth * BitmapHeight;
    uint32_t DisplayTotalSize = BitmapTotalSize * BytesPerPixel;

    void *Display;
    cudaMallocManaged(&Display, DisplayTotalSize);

    void *TrailMap;
    cudaMalloc(&TrailMap, BitmapTotalSize * sizeof(float));

    uint32_t AgentCount = 1000000;
    Agent *Agents;
    cudaMalloc(&Agents, AgentCount * sizeof(Agent));

    uint32_t AOT = 1024;
    uint32_t DisplayAOB = CalcAOB(BitmapTotalSize, AOT);
    uint32_t AgentsAOB = CalcAOB(AgentCount, AOT);

    BITMAPINFO BitmapInfo;
    BitmapInfo.bmiHeader.biSize = sizeof(BitmapInfo.bmiHeader);
    BitmapInfo.bmiHeader.biWidth = BitmapWidth;
    BitmapInfo.bmiHeader.biHeight = -BitmapHeight;
    BitmapInfo.bmiHeader.biPlanes = 1;
    BitmapInfo.bmiHeader.biBitCount = 32;
    BitmapInfo.bmiHeader.biCompression = BI_RGB;

    HDC hdc = GetDC(Window);

    UDimension2D DisplayDimension = {
        BitmapWidth,
        BitmapHeight
    };

    InitAgents<<<AgentsAOB, AOT>>>(Agents, DisplayDimension, AgentCount);
    cudaDeviceSynchronize();
    

    float DeltaTime = 0.0f;

    MSG msg = { 0 };
    int32_t running = 1;
    while (running) {
        clock_t start = clock();

        while (PeekMessageW(&msg, NULL, 0, 0, PM_REMOVE)) {
            switch (msg.message) {
                case WM_QUIT: {
                    running = 0;
                    break;
                }
            }
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }

        Render<<<AgentsAOB, AOT>>>(Agents, AgentCount, TrailMap, DisplayDimension);
        cudaDeviceSynchronize();
        UpdateAgents<<<AgentsAOB, AOT>>>(DeltaTime, Agents, AgentCount, TrailMap, DisplayDimension);
        cudaDeviceSynchronize();
        ProcessTrailMap<<<DisplayAOB, AOT>>>(DeltaTime, TrailMap, DisplayDimension);
        cudaDeviceSynchronize();
        RenderTrailMap<<<DisplayAOB, AOT>>>(TrailMap, Display, DisplayDimension);
        cudaDeviceSynchronize();
        
        StretchDIBits(
            hdc, 0, 0,
            BitmapWidth, BitmapHeight,
            0, 0,
            BitmapWidth, BitmapHeight,
            Display, &BitmapInfo,
            DIB_RGB_COLORS,
            SRCCOPY
        );

        DeltaTime = (float)(clock() - start) / CLOCKS_PER_SEC;
        // printf("fps = %f\n", 1.0f / DeltaTime);
    }

    cudaFree(Display);
    cudaFree(Agents);
    cudaFree(TrailMap);
    return 0;
}

LRESULT CALLBACK WinProcedure(HWND HWnd, UINT UMsg, WPARAM WParam, LPARAM LParam)
{
    switch (UMsg) {
        case WM_DESTROY: {
            PostQuitMessage(0);
            break;
        }
        default: {
            return DefWindowProcW(HWnd, UMsg, WParam, LParam);
            break;
        }
    }
    return 0;
}