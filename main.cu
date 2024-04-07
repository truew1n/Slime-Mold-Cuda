#include <stdio.h>
#include <windows.h>
#include <chrono>
#include <d3d11.h>

#define ERROR_REGISTER_FAILED -2

typedef struct cell_t {
    double a;
    double b;
} cell_t;


uint32_t CalcAOB(uint32_t Value, uint32_t AOT)
{
    return (Value / AOT) + ((Value % AOT) > 0);
}

#define __multi__ __host__ __device__

__multi__ cell_t Laplace2D(cell_t *Current, uint32_t LinearIndex, uint32_t Width)
{
    cell_t LaplaceCell = {0.0f, 0.0f};
    
    LaplaceCell.a += Current[LinearIndex].a * -1;
    LaplaceCell.a += Current[LinearIndex-1].a * .2;
    LaplaceCell.a += Current[LinearIndex+1].a * .2;
    LaplaceCell.a += Current[LinearIndex+Width].a * .2;
    LaplaceCell.a += Current[LinearIndex-Width].a * .2;
    LaplaceCell.a += Current[LinearIndex-Width-1].a * .05;
    LaplaceCell.a += Current[LinearIndex-Width+1].a * .05;
    LaplaceCell.a += Current[LinearIndex+Width+1].a * .05;
    LaplaceCell.a += Current[LinearIndex+Width-1].a * .05;

    LaplaceCell.b += Current[LinearIndex].b * -1;
    LaplaceCell.b += Current[LinearIndex-1].b * .2;
    LaplaceCell.b += Current[LinearIndex+1].b * .2;
    LaplaceCell.b += Current[LinearIndex+Width].b * .2;
    LaplaceCell.b += Current[LinearIndex-Width].b * .2;
    LaplaceCell.b += Current[LinearIndex-Width-1].b * .05;
    LaplaceCell.b += Current[LinearIndex-Width+1].b * .05;
    LaplaceCell.b += Current[LinearIndex+Width+1].b * .05;
    LaplaceCell.b += Current[LinearIndex+Width-1].b * .05;

    return LaplaceCell;
}

__device__ int CurrentModel = 0;

// Belousov-Zhabotinsky Reaction
// __device__ double Da = 1.0f;
// __device__ double Db = 0.5f;
// __device__ double f = 0.055f;
// __device__ double k = 0.062f;

// Mitosis
// __device__ double Da = 1.0f;
// __device__ double Db = 0.5f;
// __device__ double f = 0.0367f;
// __device__ double k = 0.0649f;

// Don't know
// __device__ double Da = 1.0f;
// __device__ double Db = 0.5f;
// __device__ double f = 0.05364f;
// __device__ double k = 0.02247f;

// Gray-Scott Model
// __device__ double Da = 0.16f;
// __device__ double Db = 0.08f;
// __device__ double f = 0.04f;
// __device__ double k = 0.06f;

// Brusselator Model
__device__ double Da = 1.0f;
__device__ double Db = 0.5f;
__device__ double f = 0.04f;
__device__ double k = 0.06f;

// Seashells Model
// __device__ double Da = 0.21f;
// __device__ double Db = 0.11f;
// __device__ double f = 0.014f;
// __device__ double k = 0.054f;

__global__ void LaplaceDiffuse(cell_t *Current, cell_t *Next, uint32_t Height, uint32_t Width)
{
    uint32_t LinearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t X = LinearIndex % Width;
    uint32_t Y = LinearIndex / Width;

    if(X <= 0 || X >= (Width - 1) || Y <= 0 || Y >= (Height - 1)) return;

    double a = Current[LinearIndex].a;
    double b = Current[LinearIndex].b;

    cell_t LaplaceCell = Laplace2D(Current, LinearIndex, Width);

    Next[LinearIndex].a = a + ((Da * LaplaceCell.a) - (a*b*b) + (f*(1.0f-a)));
    Next[LinearIndex].b = b + ((Db * LaplaceCell.b) + (a*b*b) - (b*(k+f)));
}

__device__ uint32_t DecodeRGB(uint8_t R, uint8_t G, uint8_t B)
{
    return (R << 16) + (R << 8) + B;
}

__global__ void Render(void *Display, cell_t *Next, uint32_t TotalSize)
{
    uint32_t LinearIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if(LinearIndex >= TotalSize) return;

    double a = Next[LinearIndex].a;
    double b = Next[LinearIndex].b;
    int32_t c = (int)((a-b) * 255);
    if(c > 255) c = 255;
    else if(c < 0) c = 0;
    ((uint32_t *) Display)[LinearIndex] = DecodeRGB(c, c, c);
}

__global__ void SwapCells(cell_t *Current, cell_t *Next, uint32_t TotalSize)
{
    uint32_t LinearIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if(LinearIndex >= TotalSize) return;

    cell_t NextCell = Next[LinearIndex];
    Next[LinearIndex] = Current[LinearIndex];
    Current[LinearIndex] = NextCell;
}

__global__ void Init(cell_t *Current, cell_t *Next, uint32_t Height, uint32_t Width)
{
    uint32_t LinearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t X = LinearIndex % Width;
    uint32_t Y = LinearIndex / Width;

    if(X <= 0 || X >= (Width - 1) || Y <= 0 || Y >= (Height - 1)) return;

    Current[LinearIndex].a = 1.0f;
    Current[LinearIndex].b = 0.0f;
    Next[LinearIndex].a = 1.0f;
    Next[LinearIndex].b = 0.0f;

    if(X > ((Width / 2) - 25) && X < ((Width / 2) + 25) && Y > ((Height / 2) - 25) && Y < ((Height / 2) + 25)) {
        Current[LinearIndex].b = 1.0f;
        Next[LinearIndex].b = 1.0f;
    }

    if(X > ((Width / 2) - 225) && X < ((Width / 2) - 200) && Y > ((Height / 2) - 25) && Y < ((Height / 2) + 25)) {
        Current[LinearIndex].b = 1.0f;
        Next[LinearIndex].b = 1.0f;
    }
}

LRESULT CALLBACK WinProcedure(HWND HWnd, UINT UMsg, WPARAM WParam, LPARAM LParam);

int main(void)
{
    HINSTANCE WinInstance = GetModuleHandleW(NULL);

    WNDCLASSW WinClass = {0};
    WinClass.lpszClassName = L"Reaction-Diffusion";
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
        L"Reaction Diffusion Visualization",
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
    uint32_t CellSize = sizeof(cell_t);

    uint32_t BitmapTotalSize = BitmapWidth * BitmapHeight;
    uint32_t DisplayTotalSize = BitmapTotalSize * BytesPerPixel;
    uint32_t CellsTotalSize = BitmapTotalSize * CellSize;

    void *Display;
    cell_t *Current;
    cell_t *Next;

    cudaMallocManaged(&Display, DisplayTotalSize);
    cudaMalloc(&Current, CellsTotalSize);
    cudaMalloc(&Next, CellsTotalSize);

    uint32_t AOT = 1024;
    uint32_t AOB = CalcAOB(BitmapTotalSize, AOT);

    BITMAPINFO BitmapInfo;
    BitmapInfo.bmiHeader.biSize = sizeof(BitmapInfo.bmiHeader);
    BitmapInfo.bmiHeader.biWidth = BitmapWidth;
    BitmapInfo.bmiHeader.biHeight = -BitmapHeight;
    BitmapInfo.bmiHeader.biPlanes = 1;
    BitmapInfo.bmiHeader.biBitCount = 32;
    BitmapInfo.bmiHeader.biCompression = BI_RGB;

    HDC hdc = GetDC(Window);

    Init<<<AOB, AOT>>>(Current, Next, BitmapHeight, BitmapWidth);

    uint32_t Counter = 0;
    uint32_t CounterLimit = 20;

    MSG msg = { 0 };
    int32_t running = 1;
    while (running) {

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

        LaplaceDiffuse<<<AOB, AOT>>>(Current, Next, BitmapHeight, BitmapWidth);
        cudaDeviceSynchronize();
        Render<<<AOB, AOT>>>(Display, Next, BitmapTotalSize);
        cudaDeviceSynchronize();
        SwapCells<<<AOB, AOT>>>(Current, Next, BitmapTotalSize);
        cudaDeviceSynchronize();

        if(Counter == CounterLimit) {
            StretchDIBits(
                hdc, 0, 0,
                BitmapWidth, BitmapHeight,
                0, 0,
                BitmapWidth, BitmapHeight,
                Display, &BitmapInfo,
                DIB_RGB_COLORS,
                SRCCOPY
            );
            Counter = 0;
        }
        Counter++;
    }

    cudaFree(Display);
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