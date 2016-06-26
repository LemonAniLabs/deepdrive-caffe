// This code is a mixture of http://www.directxtutorial.com/ and OBS's shared texture capture technique
// with some modifications for streaming the depth buffer and transferring shared textures efficiently with Cuda
// to Caffe's MemoryDataLayer.

//#define USE_CUDA_COPY

// include the basic windows header files and the Direct3D header files
#include <windows.h>
#include <windowsx.h>
#include <d3d11.h>
#include <d3dx11.h>
#include <d3dcompiler.h>
#include <d3dx10.h>
#include <d3d10_1.h>

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include "helper_cuda.h"

#include <iostream>
#include <sstream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cuda_runtime_api.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include "deep_drive/Agent.h"
#include "deep_drive/deep_drive.h"
#include <codecvt>
#include <locale>

#define DBOUT( s )                           \
{                                            \
   std::wostringstream os_;                  \
   os_ << s;                                 \
   OutputDebugStringW( os_.str().c_str() );  \
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#include <future>

using namespace deep_drive;

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// include the Direct3D Library file
#pragma comment (lib, "d3d11.lib")
#pragma comment (lib, "d3dx11.lib")
#pragma comment (lib, "d3d10_1.lib")
#pragma comment (lib, "d3dx10.lib")

// define the screen resolution
#define SCREEN_WIDTH  227
#define SCREEN_HEIGHT 227
#include <string>
#define SafeRelease(var) if(var) {var->Release(); var = NULL;}


// global declarations
IDXGISwapChain* swapchain; // the pointer to the swap chain interface
ID3D11Device* dev; // the pointer to our Direct3D device interface
ID3D10Device1* dev10; // the pointer to our Direct3D 10 device interface
ID3D11DeviceContext* devcon; // the pointer to our Direct3D device context
ID3D11RenderTargetView* backbuffer; // the pointer to our back buffer
ID3D11DepthStencilView* zbuffer; // the pointer to our depth buffer
ID3D11InputLayout* pLayout; // the pointer to the input layout
ID3D11VertexShader* pVS; // the pointer to the vertex shader
ID3D11PixelShader* pPS; // the pointer to the pixel shader
ID3D11Buffer* pVBuffer; // the pointer to the vertex buffer
ID3D11Buffer* pIBuffer; // the pointer to the index buffer
ID3D11Buffer* pCBuffer; // the pointer to the constant buffer
ID3D11Texture2D* depthTex;
ID3D11Texture2D* cameraTex;
ID3D11ShaderResourceView* depthView; // the pointer to the texture
ID3D11ShaderResourceView* cameraView; // the pointer to the texture
ID3D11Texture2D* backBufferTex;
HRESULT hr;
D3D11_TEXTURE2D_DESC d3dCopyDesc;
bool shouldInitCudaCopy = true;
D3D11_TEXTURE2D_DESC origDescription;
unsigned int ulLineBytes;
unsigned int screenSize;
int gameWidth;
int gameHeight;
D3D11_TEXTURE2D_DESC cpuReadDesc;
ID3D11Texture2D* pNewTexture = NULL;
ID3D11Texture2D* pCpuReadTexture = NULL;
cudaGraphicsResource* mCudaGraphicsResource;
cudaStream_t cuda_stream = NULL;
cudaArray *cuArray;


// a struct to define a single vertex
struct VERTEX
{
	FLOAT X, Y, Z;
	D3DXVECTOR3 Normal;
	FLOAT U, V;
};

// a struct to define the constant buffer
struct CBUFFER
{
	D3DXMATRIX Final;
	D3DXMATRIX Rotation;
	D3DXVECTOR4 LightVector;
	D3DXCOLOR LightColor;
	D3DXCOLOR AmbientColor;
};

// function prototypes
void InitD3D(HWND hWnd); // sets up and initializes Direct3D
void RenderFrame(void); // renders a single frame
void Cleanup(void); // closes Direct3D and releases memory
void InitGraphics(void); // creates the shape to render
void InitPipeline(void); // loads and prepares the shaders
bool copyTextureToMemory(ID3D11Texture2D* tex, BYTE *& imcopy);
void copyBackBufferToMemory(BYTE *& imcopy);
bool cudaCopyBackBufferToMemory(ID3D11Texture2D* pSurface, BYTE *& imcopy);
cv::Mat* get_screen();
void showImage(cv::Mat img);

double previousDistance = std::numeric_limits<double>::min();
double previouslyTraveled = std::numeric_limits<double>::min();

// the WindowProc function prototype
LRESULT CALLBACK WindowProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

ID3D11Texture2D* cq_CreateTextureFromSharedTextureHandle(unsigned int width, unsigned int height, HANDLE handle);
void CreateFromSharedHandle(HANDLE handle);

// Sensor shared memory
#define SHARED_CPU_MEMORY TEXT("Local\\Game2SensorMemory")
struct SharedTexData
{
	LONGLONG frameTime;
	DWORD texHandle;
	DWORD depthTexHandle;
};
SharedTexData* texData;
LPBYTE sharedMemory;
bool GetCaptureInfo(SharedTexData& ci);

// end sensor shared memory

// Agent control shared memory
HANDLE agentControlFileMap;
LPBYTE lpAgentControlSharedMemory = NULL;

void InitializeSharedAgentControlMemory(SharedAgentControlData **agentControlData)
{
    int totalSize = sizeof(SharedAgentControlData);

	std::wstringstream strName;
    strName << AGENT_CONTROL_SHARED_MEMORY;
    agentControlFileMap = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, totalSize, strName.str().c_str());
    if(!agentControlFileMap)
    {
	    return;
    }

    lpAgentControlSharedMemory = (LPBYTE)MapViewOfFile(agentControlFileMap, FILE_MAP_ALL_ACCESS, 0, 0, totalSize);
    if(!lpAgentControlSharedMemory)
    {
        CloseHandle(agentControlFileMap);
        agentControlFileMap = NULL;
		return;
    }

    *agentControlData = reinterpret_cast<SharedAgentControlData*>(lpAgentControlSharedMemory);
	(**agentControlData).should_agent_wait = true;
	(**agentControlData).heading_achieved = true;
	(**agentControlData).speed_achieved = true;

//    (*agentControlData)->frameTime = 0;

}

void DestroyAgentControlSharedMemory()
{
    if(lpAgentControlSharedMemory && agentControlFileMap)
    {
        UnmapViewOfFile(lpAgentControlSharedMemory);
        CloseHandle(agentControlFileMap);

        agentControlFileMap = NULL;
        lpAgentControlSharedMemory = NULL;
    }
}

// end agent control shared memory

// Reward shared memory

SharedRewardData* get_shared_reward_data()
{
	// Get shared CPU memory with pointer to shared GPU memory
	HANDLE hFileMap = OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, REWARD_SHARED_MEMORY);
	if (hFileMap == NULL)
	{
		return nullptr;
	}

	// Cast to our shared struct
	SharedRewardData* infoIn = static_cast<SharedRewardData*>(MapViewOfFile(
		hFileMap, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(SharedRewardData)));

	if (!infoIn)
	{
		CloseHandle(hFileMap);
		return nullptr;
	}

	return infoIn;
}

// end shared reward data

float get_reward(SharedRewardData* rewardData)
{
	double distance = double((*rewardData).distance);
	bool on_road = (*rewardData).on_road;
	if(previousDistance == std::numeric_limits<double>::min())
	{
		// First measurement
		previousDistance = distance;
		return 0;
	}
	float reward = 0.0;
	double traveled = previousDistance - distance;

	if(on_road && traveled > 0)
	{
		reward = 0.7; // Onward.
	}
	else if(on_road && traveled == 0)
	{
		reward = 0; // Waiting for something to pass like a train or traffic in intersection.
	}
	else if(on_road && traveled < 0)
	{
		reward = -1; // Distance is on-road so no need to go back.
	}
	else if( ! on_road && traveled > 0)
	{
		reward = 0.5; // Get back on the road.
	}
	else if( ! on_road && traveled < 0)
	{
		reward = 0; // Perhaps going around some off-road obstacle, don't give negative reward.
	}
	else if( ! on_road && traveled == 0 )
	{
		reward = -0.5; // No reason to stop when you're off-road. Need to non-zero travel to get back on.
	}

	if(traveled > previouslyTraveled)
	{
		// We're speeding up.
		reward += 0.3;
	}

	// Max reward: 1
	// Min reward -1
 
	previousDistance = distance;
	previouslyTraveled = traveled;
	return reward;
}

void load_pretrained_net(deep_drive::Agent*& agent_net, std::string path)
{
	bool weights_loaded = false;
	while(!weights_loaded)
	{
		try
		{
			agent_net->load_weights(path);
			weights_loaded = true;
		}
		catch(...)
		{
			LOG(ERROR) << "Could not load weights, set breakpoint here, change prototxt and retry";
			std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		}
	}
}

void InitAgent(deep_drive::Agent*& agent, SharedAgentControlData* shared_agent_control, 
	SharedRewardData* shared_reward)
{
	// Agent settings
	auto replay_memory                     = 75;    // Frames are 152k each, so this adds up
	auto replay_chance                     = 1.0;   // 0.01;  // Diversify training data with undersampling and try to approximate 1M contiguous frames as in DQN 
	auto purge_every                       = 4;     // How often to trim replay memory to max size
	auto learn_minibatch_size              = 64;    // Dynamically set batch size for "sleep" phase of 'online' training, deep_drive_model.prototxt has value of 1 set for perception
	auto num_output                        = 6;     // Also in model proto in target and fctop layers, and filled in Agent::Forward and Agent::ActuallyLearn
	auto train_iter                        = 75;    // Train every x iterations - callabrated with reload game time, but no longer necessary
	auto should_train                      = false;
	auto should_skip_update                = true;  // Hack to use train net for testing / logging outputs
	auto should_train_async                = false; // Original attempt to train and pereceive in parallel, not fully implemented yet.
	auto debug_info                        = false; // Also in solver for debug info on the backward pass
	auto should_fill_replay_memory         = true;

	auto should_resume_deep_drive          = false;
	auto resume_solver_path                = "caffe_deep_drive_train_iter_166000.solverstate";
	
	auto should_load_imagenet_pretrained   = false;
	auto weight_path_image_net             = "examples/deep_drive/bvlc_reference_caffenet.caffemodel";

	auto should_load_deep_drive_pretrained = true;
	auto weight_path_deep_drive            = "caffe_deep_drive_train_iter_206798.caffemodel";
	
	if(should_resume_deep_drive + should_load_imagenet_pretrained + should_load_deep_drive_pretrained != 1)
	{
		LOG(INFO) << "Pretrain flags are mutually exclusive";
		throw std::invalid_argument( "Pretrain flags are mutually exclusive" );
	}

	if( ! should_resume_deep_drive)
	{
		resume_solver_path = "";
	}

	agent = new deep_drive::Agent(replay_memory, learn_minibatch_size, num_output,
		"examples/deep_drive/deep_drive_solver.prototxt", 
		"examples/deep_drive/deep_drive_model.prototxt",
		should_train, train_iter,
		should_train_async, 
		shared_agent_control, shared_reward, resume_solver_path, debug_info,
		replay_chance, purge_every, should_skip_update, should_fill_replay_memory);
	
	// Fine tune
	if(should_load_deep_drive_pretrained)
	{
		load_pretrained_net(agent, weight_path_deep_drive);
	}	
	else if(should_load_imagenet_pretrained)
	{
		load_pretrained_net(agent, weight_path_image_net);
	}
}

SharedAgentControlData* wait_for_auto_it_sharing()
{
	SharedAgentControlData* agent_control_data;
	InitializeSharedAgentControlMemory(&agent_control_data);

	while ((*agent_control_data).should_agent_wait) {
		output("Waiting for AutoIt to respond...");
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
	}
	return agent_control_data;
}

SharedRewardData* wait_for_shared_reward_data()
{
	SharedRewardData* shared_reward_data = nullptr;
	while (shared_reward_data == nullptr) {
		output("Trying to access shared reward data...");
		shared_reward_data = get_shared_reward_data();
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
	}
	return shared_reward_data;
}

double denormalize_speed(double speed)
{
	return speed / kSpeedCoefficient;
}

// the entry point for any Windows program
int WINAPI WinMain(HINSTANCE hInstance,
                   HINSTANCE hPrevInstance,
                   LPSTR lpCmdLine,
                   int nCmdShow)
{
	// Logs are in user folder AppData/Local/temp
	google::InitGoogleLogging("caffe.exe");

	auto shared_agent_data = wait_for_auto_it_sharing();
	auto shared_reward_data = wait_for_shared_reward_data();

	HWND hWnd;
	WNDCLASSEX wc;

	ZeroMemory(&wc, sizeof(WNDCLASSEX));

	wc.cbSize = sizeof(WNDCLASSEX);
	wc.style = CS_HREDRAW | CS_VREDRAW;
	wc.lpfnWndProc = WindowProc;
	wc.hInstance = hInstance;
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);
	wc.lpszClassName = L"WindowClass";

	RegisterClassEx(&wc);

	RECT wr = {0, 0, SCREEN_WIDTH, SCREEN_HEIGHT};
	AdjustWindowRect(&wr, WS_OVERLAPPEDWINDOW, FALSE);

	hWnd = CreateWindowEx(NULL,
	                          L"WindowClass",
	                          L"Game2Sensor",
	                          WS_OVERLAPPEDWINDOW,
	                          300,
	                          300,
	                          wr.right - wr.left,
	                          wr.bottom - wr.top,
	                          NULL,
	                          NULL,
	                          hInstance,
	                          NULL);

	ShowWindow(hWnd, nCmdShow);

	// set up and initialize Direct3D
	InitD3D(hWnd);

	cv::Mat* blank = get_screen(); // First screen is always black.

	bool should_show_image = false;
	bool should_skip_agent = false;

	Agent* agent = nullptr;

	if( ! should_skip_agent)
	{
		InitAgent(agent, shared_agent_data, shared_reward_data);
	}

//	int action = 0;
//	int current_action = 0;
	// TODO: Put these in config file
	int step = 0;
	(*shared_agent_data).step = step;
	bool should_save_data = true;
	int save_input_every = 1;
	bool should_reload_based_on_distance = false;
	bool should_reload_based_on_duration = true;
	bool should_deep_drive_only = false;
	bool should_toggle_game_and_deep_drive = false;
	shared_reward_data->should_perform_random_action = true; // Otherwise, we just mildly accelerate and disengage
	auto kMsToExploitOrExplore = 1000;
	auto kStepsToExploitOrExplore = kMsToExploitOrExplore / kMsInStep;
	auto kStepsInExploitExploreCycle = 2 * kStepsToExploitOrExplore;

	auto reload_game_after = std::chrono::minutes(20);

	double last_speed = 0;
//	bool manual_action = true;
	double accumulated_spin = 0;

	std::chrono::system_clock::time_point game_start_time = 
		reload_game(shared_agent_data, shared_reward_data, should_save_data);

	// Main loop
	MSG msg;
	while (true)
	{
		auto start_time = std::chrono::high_resolution_clock::now();
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);

			if (msg.message == WM_QUIT)
				break;
		}

 		RenderFrame();

		auto screen_cap_start = std::chrono::high_resolution_clock::now();
		cv::Mat* screen = get_screen();
//		print_time_since(screen_cap_start, "screen cap");
		auto screen_cap_end   = std::chrono::high_resolution_clock::now();

		if(screen)
		{
			if(should_show_image)
			{
				showImage(*screen);
			}
		}

		if(step % 100 == 0)
		{
			reset_game_mod_options(shared_reward_data);
		}

		if(screen == nullptr)
		{ 
			output("Could not get screen, skipping..");
		}
		else if(should_skip_agent)
		{
			output("should_skip_agent set, skipping dqn...");
			delete screen;
		}
		else if((*shared_agent_data).should_agent_wait)
		{
			output("Perception paused, skipping...");
			if(should_save_data)
			{
				shared_reward_data->should_game_drive = true;
			}

			delete screen;
		}
		else
		{
			if(step == 0) // Let heading and speed initialize.
			{
				last_speed = shared_reward_data->speed;
			}
			else {
				if(should_deep_drive_only || (should_toggle_game_and_deep_drive && step % kStepsInExploitExploreCycle < kStepsToExploitOrExplore))
				{
					shared_reward_data->should_game_drive = false;
				}
				else if(shared_reward_data->should_perform_random_action && get_random_double(0, 1) < 0.10)
				{
					shared_reward_data->should_game_drive = false;
					std::this_thread::sleep_for(std::chrono::milliseconds(1000));
				}
				else
				{

					shared_reward_data->should_game_drive = true;
				}

				double speed_change = shared_reward_data->speed - last_speed;
				last_speed = shared_reward_data->speed;
//				current_action = agent->infer_action(shared_reward_data->rotational_velocity, speed_change);
				Action next_action = agent->Perceive(screen, shared_reward_data->spin, 
					shared_reward_data->speed, speed_change, shared_agent_data->steer, 
					shared_agent_data->throttle);
//				print_time_since(screen_cap_start, "screen cap to perceive");
				accumulated_spin += next_action.spin;
				if(agent->get_should_train())
				{
					// We are not controlling acceleration.
					(*shared_agent_data).heading_achieved = true;
					(*shared_agent_data).speed_achieved = true;
				}
				else
				{

//					double current_speed = normalize_speed(shared_reward_data->speed);

//					double test_vjoy_spin_temp = 0.0;
//					(*shared_agent_data).action = agent->infer_action(test_vjoy_spin_temp, 10.0, shared_reward_data->speed);

					if(should_save_data)
					{
						saveInput(shared_reward_data, shared_agent_data, 
							step, should_save_data, save_input_every, screen, screen_cap_start);					
					}

					(*shared_agent_data).heading_achieved = true;
					(*shared_agent_data).speed_achieved = true;						

					(*shared_reward_data).desired_spin                 = next_action.spin;
					(*shared_agent_data ).desired_spin                 = next_action.spin;
					(*shared_reward_data).desired_speed                = denormalize_speed(next_action.speed); 
					(*shared_agent_data ).desired_speed                = denormalize_speed(next_action.speed); 
					(*shared_reward_data).desired_speed_change         = next_action.speed_change;
					(*shared_agent_data ).desired_speed_change         = next_action.speed_change;
					(*shared_reward_data).desired_direction            = next_action.direction;
					(*shared_agent_data ).desired_direction            = next_action.direction;

					(*shared_agent_data ).desired_steer                = next_action.steer;
					(*shared_agent_data ).desired_throttle             = next_action.throttle;						


					while(time_left_in_step(start_time))
					{
						// This has no effect on saved data.
						// It just allows the agent to react to the state of the vehicle throughout the duration
						// of the step.
						(*shared_agent_data).actual_spin         = shared_reward_data->spin;
						(*shared_agent_data).actual_speed        = shared_reward_data->speed;
						(*shared_agent_data).actual_speed_change = shared_reward_data->speed - last_speed;						
					}

//					// Perform a random action once in a while
//					if(should_save_data && get_random_double(0, 1) < 0.50)
//					{
//						shared_reward_data->should_wander = false;
//						shared_reward_data->is_wandering = false;
//						std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // Give time to get off track.
//						shared_reward_data->should_wander = true;
////						std::this_thread::sleep_for(std::chrono::milliseconds(230)); // Give time to take back control.
//					}

					if( ! agent->get_should_save_experiences())
					{
						(*screen).release();
						delete screen;
					}

				}
			}
		}

		if (should_reload_based_on_distance)
		{
			if( shared_reward_data->distance <= 10)
			{
				should_create_saved_input_folder = true;
				game_start_time = reload_game(shared_agent_data, shared_reward_data, should_save_data);
				step = 0;				
			}
			else
			{
				step++;
			}

		} else if(should_reload_based_on_duration)
		{
			game_start_time = reload_game_based_on_duration(reload_game_after, game_start_time, shared_agent_data, shared_reward_data, should_save_data, step);
		} else
		{
			step++;
		}

		(*shared_agent_data).step = step;

		enforce_period(start_time);

	}

	delete shared_reward_data;
	delete shared_agent_data;

	// clean up DirectX and COM
	Cleanup();

	return msg.wParam;
}


// this is the main message handler for the program
LRESULT CALLBACK WindowProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message)
	{
	case WM_DESTROY:
		{
			PostQuitMessage(0);
			return 0;
		}
		break;
	}

	return DefWindowProc(hWnd, message, wParam, lParam);
}

UINT creationFlags = NULL;


// this function initializes and prepares Direct3D for use
void InitD3D(HWND hWnd)
{
	// create a struct to hold information about the swap chain
	DXGI_SWAP_CHAIN_DESC scd;

#if defined(_DEBUG)
	// If the project is in a debug build, enable the debug layer.
	creationFlags = D3D11_CREATE_DEVICE_DEBUG;
#endif

	// clear out the struct for use
	ZeroMemory(&scd, sizeof(DXGI_SWAP_CHAIN_DESC));

	// fill the swap chain description struct
	scd.BufferCount = 1; // one back buffer
	scd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM; // use 32-bit color
	scd.BufferDesc.Width = SCREEN_WIDTH; // set the back buffer width
	scd.BufferDesc.Height = SCREEN_HEIGHT; // set the back buffer height
	scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT; // how swap chain is to be used
	scd.OutputWindow = hWnd; // the window to be used
	scd.SampleDesc.Count = 4; // how many multisamples
	scd.Windowed = TRUE; // windowed/full-screen mode
	scd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH; // allow full-screen switching

	D3D_FEATURE_LEVEL featureLevels[] =
	{
		D3D_FEATURE_LEVEL_10_1,
	};

	D3D_FEATURE_LEVEL m_featureLevel;

	// create a device, device context and swap chain using the information in the scd struct
	hr = D3D11CreateDeviceAndSwapChain(NULL,
	                              D3D_DRIVER_TYPE_HARDWARE,
	                              NULL,
	                              creationFlags,
	                              featureLevels,
	                              ARRAYSIZE(featureLevels),
	                              D3D11_SDK_VERSION,
	                              &scd,
	                              &swapchain,
	                              &dev,
	                              &m_featureLevel,
	                              &devcon);



	// create the depth buffer texture
	D3D11_TEXTURE2D_DESC texd;
	ZeroMemory(&texd, sizeof(texd));

	texd.Width = SCREEN_WIDTH;
	texd.Height = SCREEN_HEIGHT;
	texd.ArraySize = 1;
	texd.MipLevels = 1;
	texd.SampleDesc.Count = 4;
	texd.Format = DXGI_FORMAT_D32_FLOAT;
	texd.BindFlags = D3D11_BIND_DEPTH_STENCIL;

	ID3D11Texture2D* pDepthBuffer;
	dev->CreateTexture2D(&texd, NULL, &pDepthBuffer);

	// create the depth buffer
	D3D11_DEPTH_STENCIL_VIEW_DESC dsvd;
	ZeroMemory(&dsvd, sizeof(dsvd));

	dsvd.Format = DXGI_FORMAT_D32_FLOAT;
	dsvd.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DMS;

	dev->CreateDepthStencilView(pDepthBuffer, &dsvd, &zbuffer);
	pDepthBuffer->Release();

	// get the address of the back buffer
	ID3D11Texture2D* pBackBuffer;
	swapchain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&pBackBuffer);

	// use the back buffer address to create the render target
	dev->CreateRenderTargetView(pBackBuffer, NULL, &backbuffer);
	pBackBuffer->Release();

	// set the render target as the back buffer
	devcon->OMSetRenderTargets(1, &backbuffer, zbuffer);


	// Set the viewport
	D3D11_VIEWPORT viewport;
	ZeroMemory(&viewport, sizeof(D3D11_VIEWPORT));

	viewport.TopLeftX = 0;
	viewport.TopLeftY = 0;
	viewport.Width = SCREEN_WIDTH;
	viewport.Height = SCREEN_HEIGHT;
	viewport.MinDepth = 0; // the closest an object can be on the depth buffer is 0.0
	viewport.MaxDepth = 1; // the farthest an object can be on the depth buffer is 1.0

	devcon->RSSetViewports(1, &viewport);

	InitPipeline();
	InitGraphics();

	SharedTexData info;
	while (GetCaptureInfo(info) == false)
	{
		DBOUT("********"
			<< "Shared memory not available, have you changed resolution to trigger shared memory?" 
			<< "********"
			<< std::endl);
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		//throw std::runtime_error("Shared memory not available, have you changed resolution to trigger shared memory?");
	}
}

static float Z_ROTATE = 1.5f * M_PI;

// this is the function used to render a single frame
void RenderFrame(void)
{
	CBUFFER cBuffer;

	cBuffer.LightVector = D3DXVECTOR4(1.0f, 1.0f, 1.0f, 0.0f);
	cBuffer.LightColor = D3DXCOLOR(0.5f, 0.5f, 0.5f, 1.0f);
	cBuffer.AmbientColor = D3DXCOLOR(0.2f, 0.2f, 0.2f, 1.0f);

	D3DXMATRIX matRotate, matView, matProjection;
	D3DXMATRIX matFinal;

	D3DXMatrixRotationZ(&matRotate, Z_ROTATE);

	// create a view matrix
	D3DXMatrixLookAtLH(&matView,
	                   &D3DXVECTOR3(0.0f, 0.0f, 3.3f), // the camera position
	                   &D3DXVECTOR3(0.0f, 0.0f, 0.0f), // the look-at position
	                   &D3DXVECTOR3(0.0f, 1.0f, 0.0f)); // the up direction

	// create a projection matrix
	D3DXMatrixPerspectiveFovLH(&matProjection,
	                           (FLOAT)D3DXToRadian(45), // field of view
	                           (FLOAT)SCREEN_WIDTH / (FLOAT)SCREEN_HEIGHT, // aspect ratio
	                           1.0f, // near view-plane
	                           100.0f); // far view-plane

	// load the matrices into the constant buffer
	cBuffer.Final = matRotate * matView * matProjection;
	cBuffer.Rotation = matRotate;

	// clear the back buffer to a deep blue
	devcon->ClearRenderTargetView(backbuffer, D3DXCOLOR(0.0f, 0.2f, 0.4f, 1.0f));

	// clear the depth buffer
	devcon->ClearDepthStencilView(zbuffer, D3D11_CLEAR_DEPTH, 1.0f, 0);

	// select which vertex buffer to display
	UINT stride = sizeof(VERTEX);
	UINT offset = 0;
	devcon->IASetVertexBuffers(0, 1, &pVBuffer, &stride, &offset);
	devcon->IASetIndexBuffer(pIBuffer, DXGI_FORMAT_R32_UINT, 0);

	// select which primtive type we are using
	devcon->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	// draw the Hypercraft
	devcon->UpdateSubresource(pCBuffer, 0, 0, &cBuffer, 0, 0);

//	//////------//-/-/-/-/-/-/-/-/-/-/-/ Move this back to init for depth rendering
//
//  // Create a resource view for the depth texture. Camera texture is already moved out of render.
//
//	HRESULT hr;
//	ID3D11ShaderResourceView* pSRV = NULL;
//
//	D3D11_TEXTURE2D_DESC TexDesc;
//	depthTex->GetDesc(&TexDesc);
//
//	D3D11_SHADER_RESOURCE_VIEW_DESC SRVDesc;
//	::ZeroMemory(&SRVDesc,sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC));
//	SRVDesc.Texture2D.MipLevels = TexDesc.MipLevels;
//	SRVDesc.Texture2D.MostDetailedMip = 0;
//	SRVDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
//	SRVDesc.Format = TexDesc.Format;
//
//	hr = dev->CreateShaderResourceView(cameraTex, &SRVDesc, &cameraView);

//	//////------//-/-/-/-/-/-/-/-/-/-/-/

	devcon->PSSetShaderResources(0, 1, &cameraView);
	devcon->DrawIndexed(36, 0, 0);

	// switch the back buffer and the front buffer
	swapchain->Present(0, 0);
}

cv::Mat* get_screen()
{
	HRESULT hr = swapchain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&backBufferTex));
	BYTE * imcopy = nullptr;
	bool success;
	bool use_cuda = true;

	auto screen_copy_start = std::chrono::high_resolution_clock::now();
	if(use_cuda)
	{
		success = cudaCopyBackBufferToMemory(backBufferTex, imcopy);
	}
	else
	{
		success = copyTextureToMemory(backBufferTex, imcopy);
	}
//	print_time_since(screen_copy_start, "screen gpu transfer");

	backBufferTex->Release();

	if(success)
	{
		//imcopy[0] = 1; /// DEELLLLLLLLLEEEEEEETTTTTTE THISSS@

		size_t step = SCREEN_WIDTH * 4;
		cv::Mat* img_pt = new cv::Mat(SCREEN_HEIGHT, SCREEN_WIDTH, CV_8UC4, imcopy, step);
//		print_time_since(screen_copy_start, "screen new");

		cv::Mat img = *img_pt;
//		cv::Mat ret_im = img.clone();
		cv::Mat* ret_pt = new cv::Mat(SCREEN_HEIGHT, SCREEN_WIDTH, CV_8UC4);
		img_pt->copyTo(*ret_pt); // Make sure we decouple from imcopy
//		print_time_since(screen_copy_start, "screen copy");

		//	cv::Mat img = cv::imdecode(, CV_32FC4);
		cv::cvtColor(*ret_pt, *ret_pt, CV_RGBA2BGR);
//		print_time_since(screen_copy_start, "screen convert");
		//cv::Mat src = *img;
		//cv::Mat ret = src.clone(); // Give OpenCV ownership of the memory



		//src.release();
		delete img_pt;
		delete[] imcopy;
//		print_time_since(screen_copy_start, "screen delete");
		return ret_pt;
//		return img;
	} 
	else
	{
		return nullptr;
	}

}

void CreateFromSharedHandle(HANDLE handle)
{
    HRESULT err;

    if(!handle)
    {
		throw new std::runtime_error("Shared handle not set in game process");
    }

    ID3D11Resource *tempResource;
    if(FAILED(err =dev->OpenSharedResource(handle, __uuidof(ID3D11Resource), (void**)&tempResource)))
    {
        throw new std::runtime_error("Could not cast shared handle to d3d resource");
    }

    if(FAILED(err = tempResource->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&cameraTex)))
    {
        SafeRelease(tempResource);
		throw new std::runtime_error("Could not cast d3d resource to d3d texture");
    }

    tempResource->Release();

    D3D11_TEXTURE2D_DESC td;
    cameraTex->GetDesc(&td);

    D3D11_SHADER_RESOURCE_VIEW_DESC resourceDesc;
    ZeroMemory(&resourceDesc, sizeof(resourceDesc));
    resourceDesc.Format              = td.Format;
    resourceDesc.ViewDimension       = D3D11_SRV_DIMENSION_TEXTURE2D;
    resourceDesc.Texture2D.MipLevels = 1;

    ID3D11ShaderResourceView *resource = NULL;
    if(FAILED(hr = dev->CreateShaderResourceView(cameraTex, NULL, &cameraView)))
    {
        SafeRelease(cameraTex);
		throw new std::runtime_error("Failed to created resource view");
    }
}

void Cleanup(void)
{
	swapchain->SetFullscreenState(FALSE, NULL); // switch to windowed mode

	cudaFree(cuArray);
	cudaStreamDestroy(cuda_stream);
	cudaGraphicsUnregisterResource(mCudaGraphicsResource);
    getLastCudaError("cudaGraphicsUnregisterResource (g_texture_2d) failed");

	DestroyAgentControlSharedMemory();

	pCpuReadTexture->Release();
	pNewTexture->Release();
	backBufferTex->Release();
	cameraView->Release();
	depthView->Release();
	cameraTex->Release();
	depthTex->Release();
	zbuffer->Release();
	pLayout->Release();
	pVS->Release();
	pPS->Release();
	pVBuffer->Release();
	pIBuffer->Release();
	pCBuffer->Release();
	swapchain->Release();
	backbuffer->Release();

	// cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

	dev->Release();
	devcon->Release();


}

// this is the function that creates the shape to render
void InitGraphics()
{
    // TODO: Delete this. Was from directxtutorial.com boilerplate.
    //       However, could be useful for 3D visualization of net params.

	// create vertices to represent the corners of the cube
	VERTEX OurVertices[] =
		{
			{-1.0f, -1.0f, 1.0f, D3DXVECTOR3(0.0f, 0.0f, 1.0f), 0.0f, 0.0f}, // side 1
			{1.0f, -1.0f, 1.0f, D3DXVECTOR3(0.0f, 0.0f, 1.0f), 0.0f, 1.0f},
			{-1.0f, 1.0f, 1.0f, D3DXVECTOR3(0.0f, 0.0f, 1.0f), 1.0f, 0.0f},
			{1.0f, 1.0f, 1.0f, D3DXVECTOR3(0.0f, 0.0f, 1.0f), 1.0f, 1.0f},

			{-1.0f, -1.0f, -1.0f, D3DXVECTOR3(0.0f, 0.0f, -1.0f), 0.0f, 0.0f}, // side 2
			{-1.0f, 1.0f, -1.0f, D3DXVECTOR3(0.0f, 0.0f, -1.0f), 0.0f, 1.0f},
			{1.0f, -1.0f, -1.0f, D3DXVECTOR3(0.0f, 0.0f, -1.0f), 1.0f, 0.0f},
			{1.0f, 1.0f, -1.0f, D3DXVECTOR3(0.0f, 0.0f, -1.0f), 1.0f, 1.0f},

			{-1.0f, 1.0f, -1.0f, D3DXVECTOR3(0.0f, 1.0f, 0.0f), 0.0f, 0.0f}, // side 3
			{-1.0f, 1.0f, 1.0f, D3DXVECTOR3(0.0f, 1.0f, 0.0f), 0.0f, 1.0f},
			{1.0f, 1.0f, -1.0f, D3DXVECTOR3(0.0f, 1.0f, 0.0f), 1.0f, 0.0f},
			{1.0f, 1.0f, 1.0f, D3DXVECTOR3(0.0f, 1.0f, 0.0f), 1.0f, 1.0f},

			{-1.0f, -1.0f, -1.0f, D3DXVECTOR3(0.0f, -1.0f, 0.0f), 0.0f, 0.0f}, // side 4
			{1.0f, -1.0f, -1.0f, D3DXVECTOR3(0.0f, -1.0f, 0.0f), 0.0f, 1.0f},
			{-1.0f, -1.0f, 1.0f, D3DXVECTOR3(0.0f, -1.0f, 0.0f), 1.0f, 0.0f},
			{1.0f, -1.0f, 1.0f, D3DXVECTOR3(0.0f, -1.0f, 0.0f), 1.0f, 1.0f},

			{1.0f, -1.0f, -1.0f, D3DXVECTOR3(1.0f, 0.0f, 0.0f), 0.0f, 0.0f}, // side 5
			{1.0f, 1.0f, -1.0f, D3DXVECTOR3(1.0f, 0.0f, 0.0f), 0.0f, 1.0f},
			{1.0f, -1.0f, 1.0f, D3DXVECTOR3(1.0f, 0.0f, 0.0f), 1.0f, 0.0f},
			{1.0f, 1.0f, 1.0f, D3DXVECTOR3(1.0f, 0.0f, 0.0f), 1.0f, 1.0f},

			{-1.0f, -1.0f, -1.0f, D3DXVECTOR3(-1.0f, 0.0f, 0.0f), 0.0f, 0.0f}, // side 6
			{-1.0f, -1.0f, 1.0f, D3DXVECTOR3(-1.0f, 0.0f, 0.0f), 0.0f, 1.0f},
			{-1.0f, 1.0f, -1.0f, D3DXVECTOR3(-1.0f, 0.0f, 0.0f), 1.0f, 0.0f},
			{-1.0f, 1.0f, 1.0f, D3DXVECTOR3(-1.0f, 0.0f, 0.0f), 1.0f, 1.0f},
		};


	// create the vertex buffer
	D3D11_BUFFER_DESC bd;
	ZeroMemory(&bd, sizeof(bd));

	bd.Usage = D3D11_USAGE_DYNAMIC;
	bd.ByteWidth = sizeof(VERTEX) * 24;
	bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

	dev->CreateBuffer(&bd, NULL, &pVBuffer);

	// copy the vertices into the buffer
	D3D11_MAPPED_SUBRESOURCE ms;
	devcon->Map(pVBuffer, NULL, D3D11_MAP_WRITE_DISCARD, NULL, &ms); // map the buffer
	memcpy(ms.pData, OurVertices, sizeof(OurVertices)); // copy the data
	devcon->Unmap(pVBuffer, NULL);


	// create the index buffer out of DWORDs
	DWORD OurIndices[] =
		{
			0, 1, 2, // side 1
			2, 1, 3,
			4, 5, 6, // side 2
			6, 5, 7,
			8, 9, 10, // side 3
			10, 9, 11,
			12, 13, 14, // side 4
			14, 13, 15,
			16, 17, 18, // side 5
			18, 17, 19,
			20, 21, 22, // side 6
			22, 21, 23,
		};

	// create the index buffer
	bd.Usage = D3D11_USAGE_DYNAMIC;
	bd.ByteWidth = sizeof(DWORD) * 36;
	bd.BindFlags = D3D11_BIND_INDEX_BUFFER;
	bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	bd.MiscFlags = 0;

	dev->CreateBuffer(&bd, NULL, &pIBuffer);

	devcon->Map(pIBuffer, NULL, D3D11_MAP_WRITE_DISCARD, NULL, &ms); // map the buffer
	memcpy(ms.pData, OurIndices, sizeof(OurIndices)); // copy the data
	devcon->Unmap(pIBuffer, NULL);

#ifdef USE_CUDA_COPY
	HRESULT hr = cudaD3D11SetDirect3DDevice(dev);
	if (hr != S_OK){ throw "Could not bind CUDA device to DirectX device"; }
#endif
}


// this function loads and prepares the shaders
void InitPipeline()
{
	UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
#if defined( DEBUG ) || defined( _DEBUG )
	flags |= D3DCOMPILE_DEBUG;
#endif

	// compile the shaders
	ID3D10Blob *VS, *PS;
	D3DX11CompileFromFile(L"shaders.fx", 0, 0, "VShader", "vs_4_0", 0, 0, 0, &VS, 0, 0);
	D3DX11CompileFromFile(L"shaders.fx", 0, 0, "PShaderPassThrough", "ps_4_0", 0, 0, 0, &PS, 0, 0);

	// create the shader objects
	dev->CreateVertexShader(VS->GetBufferPointer(), VS->GetBufferSize(), NULL, &pVS);
	dev->CreatePixelShader(PS->GetBufferPointer(), PS->GetBufferSize(), NULL, &pPS);

	// set the shader objects
	devcon->VSSetShader(pVS, 0, 0);
	devcon->PSSetShader(pPS, 0, 0);

	// create the input element object
	D3D11_INPUT_ELEMENT_DESC ied[] =
		{
			{"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
			{"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0},
			{"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 24, D3D11_INPUT_PER_VERTEX_DATA, 0},
		};

	// use the input element descriptions to create the input layout
	dev->CreateInputLayout(ied, 3, VS->GetBufferPointer(), VS->GetBufferSize(), &pLayout);
	devcon->IASetInputLayout(pLayout);

	// create the constant buffer
	D3D11_BUFFER_DESC bd;
	ZeroMemory(&bd, sizeof(bd));

	bd.Usage = D3D11_USAGE_DEFAULT;
	bd.ByteWidth = 176;
	bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;

	dev->CreateBuffer(&bd, NULL, &pCBuffer);
	devcon->VSSetConstantBuffers(0, 1, &pCBuffer);
}


bool GetCaptureInfo(SharedTexData& ci)
{
	// Get shared CPU memory with pointer to shared GPU memory
	HANDLE hFileMap = OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, SHARED_CPU_MEMORY);
	if (hFileMap == NULL)
	{
		return false;
	}

	// Cast to our shared struct
	SharedTexData* infoIn;
	infoIn = (SharedTexData*)MapViewOfFile(hFileMap, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(SharedTexData));
	if (!infoIn)
	{
		CloseHandle(hFileMap);
		return false;
	}

	// Small copy into local struct
	memcpy(&ci, infoIn, sizeof(SharedTexData));

	HANDLE depthHandle = (HANDLE)infoIn->depthTexHandle;
	HANDLE texHandle = (HANDLE)infoIn->texHandle;

	depthTex = cq_CreateTextureFromSharedTextureHandle(SCREEN_WIDTH, SCREEN_HEIGHT, depthHandle);
	CreateFromSharedHandle(texHandle);

	if (infoIn)
	{
		UnmapViewOfFile(infoIn);
	}

	if (hFileMap)
	{
		CloseHandle(hFileMap);
	}

	return true;
}

// Currently used for depth texture only. Camera texture done in CreateFromSharedHandle, although these two
// could be merged now.
// TODO: Merge with CreateFromSharedHandle
ID3D11Texture2D* cq_CreateTextureFromSharedTextureHandle(unsigned int width, unsigned int height, HANDLE handle)
{
	HRESULT err;
	if (!handle)
	{
		return NULL;
	}

	ID3D11Resource *tempResource;
    if(FAILED(err = dev->OpenSharedResource(handle, __uuidof(ID3D11Resource), (void**)&tempResource)))
    {
        DBOUT(TEXT("cq_CreateTextureFromSharedTextureHandle: Failed to open shared handle, result = 0x%08lX"), err);
        return NULL;
    }

	ID3D11Texture2D* texVal;
	if (FAILED(err = tempResource->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&texVal)))
	{
		SafeRelease(tempResource);
		return NULL;
	}

	tempResource->Release();
	return texVal;
}

void showImage(cv::Mat img)
{
	if (! img.data) // Check for invalid input
	{
		std::cout << "Could not open or find the image" << std::endl ;
		return;
	}

//	img = img.t();

	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);// Create a window for display.
	cv::imshow("Display window", img); // Show our image inside it.

	cv::waitKey(0);
}

bool copyTextureToMemory(ID3D11Texture2D* tex, BYTE *& imcopy)
{
	ID3D11Texture2D* pNewTexture = NULL;
	D3D11_TEXTURE2D_DESC description = {
		SCREEN_WIDTH,//UINT Width;
		SCREEN_HEIGHT,//UINT Height;
		1,//UINT MipLevels;
		1,//UINT ArraySize;
		DXGI_FORMAT_R8G8B8A8_UNORM, //DXGI_FORMAT_R32G32B32A32_FLOAT,//DXGI_FORMAT Format;
		1, 0,//DXGI_SAMPLE_DESC SampleDesc;
		D3D11_USAGE_DEFAULT,// On GPU transfer
		0, //UINT BindFlags;
		0, //UINT CPUAccessFlags;
		0  //UINT MiscFlags;
	};

	HRESULT hr = dev->CreateTexture2D(&description, 0, &pNewTexture);

	if (pNewTexture) {
			devcon->ResolveSubresource(pNewTexture, 0, tex, 0, DXGI_FORMAT_R8G8B8A8_UNORM); // GUID_WICPixelFormat32bppRGBA
	} else
	{
		return false;
	}

    ID3DBlob* pBlob;
    hr = D3DX11SaveTextureToMemory(devcon, pNewTexture, D3DX11_IFF_BMP, &pBlob, NULL);
	imcopy = static_cast<unsigned char *>(pBlob->GetBufferPointer()); // data is ff 7f 33 00 for BGRA8 vs GPU is 33 7f ff 00
//	showImage(buf, pBlob->GetBufferSize());
	//std::vector<char> data(buf, buf + pBlob->GetBufferSize()); 
	return true;
}

void cudaCopyCleanup()
{
	SafeRelease(pNewTexture);
	SafeRelease(pCpuReadTexture);
	cudaFree(cuArray);
	cudaFree(cuArray);
	cudaStreamDestroy(cuda_stream);
	cudaGraphicsUnregisterResource(mCudaGraphicsResource);
	getLastCudaError("cudaGraphicsUnregisterResource (g_texture_2d) failed");
}

bool cudaFail(cudaError_t cudaStatus)
{
	if(cudaStatus != cudaSuccess)
	{
		return true;
	}
	return false;
}

bool cudaCopyBackBufferToMemory(ID3D11Texture2D* tex, BYTE *& imcopy)
{
	bool ret = true;
	D3D11_TEXTURE2D_DESC desc_test;
	tex->GetDesc(&desc_test);

	if(desc_test.SampleDesc.Count != origDescription.SampleDesc.Count && shouldInitCudaCopy == false)
	{
		// Backbuffer changed format, reinit.
		cudaCopyCleanup();
		shouldInitCudaCopy = true;
	}

	if (tex && shouldInitCudaCopy)
	{
		tex->GetDesc(&origDescription);
		gameWidth = origDescription.Width;
		gameHeight = origDescription.Height;
		ulLineBytes = gameWidth * 4; // 4 Bytes = 32 bit
		screenSize = ulLineBytes * gameHeight;

		d3dCopyDesc = {
			gameWidth,//UINT Width;
			gameHeight,//UINT Height;
			1,//UINT MipLevels;
			1,//UINT ArraySize;
			origDescription.Format, // DXGI_FORMAT_R8G8B8A8_UNORM DXGI_FORMAT
			{
				1, // Sample count must be 1 in order to not have bind flags
				0  // Quality
			}, //DXGI_SAMPLE_DESC SampleDesc;
			D3D11_USAGE_DEFAULT,// On GPU transfer
			0,//UINT BindFlags;
			0,//UINT CPUAccessFlags;
			0//UINT MiscFlags;
		};

		cpuReadDesc = {
			gameWidth,//UINT Width;
			gameHeight,//UINT Height;
			1, // UINT MipLevels;
			1, // UINT ArraySize;
			origDescription.Format, // DXGI_FORMAT_R8G8B8A8_UNORM DXGI_FORMAT
			{
				1, // A multisampled Texture2D cannot be bound to certain parts of the graphics pipeline, but must have at least one BindFlags bit set. The following BindFlags bits (0) cannot be set in this case: D3D11_BIND_VERTEX_BUFFER (0), D3D11_BIND_INDEX_BUFFER (0), D3D11_BIND_CONSTANT_BUFFER (0), D3D11_BIND_STREAM_OUTPUT (0). [ STATE_CREATION ERROR #99: CREATETEXTURE2D_INVALIDBINDFLAGS]
				0
			}, //DXGI_SAMPLE_DESC SampleDesc count, quality;
			D3D11_USAGE_STAGING,//D3D11_USAGE Usage; Allow transfer from GPU to CPU
			0,//UINT BindFlags;
			D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE, //UINT CPUAccessFlags;
			0//UINT MiscFlags;
		};

		hr = dev->CreateTexture2D(&d3dCopyDesc, 0, &pNewTexture);
		if (pNewTexture)
		{	
			hr = dev->CreateTexture2D(&cpuReadDesc, 0, &pCpuReadTexture);
		}

		if(cudaFail(cudaGraphicsD3D11RegisterResource(&mCudaGraphicsResource, pCpuReadTexture,
			cudaGraphicsMapFlagsNone)))
		{
			ret = false;
		}
		else if(cudaFail(cudaStreamCreate(&cuda_stream)))
		{
			ret = false;
		}
		else if(cudaFail(cudaGraphicsMapResources(1, &mCudaGraphicsResource, cuda_stream)))
		{
			ret = false;
		}
		else if(cudaFail(cudaGraphicsSubResourceGetMappedArray(&cuArray, mCudaGraphicsResource, 0, 0)))
		{
			ret = false;				
		}

		shouldInitCudaCopy = false;
	}

	if (pNewTexture && ret == true)
	{	
		//devcon->CopyResource(pNewTexture, tex);
		devcon->ResolveSubresource(pNewTexture, 0, tex, 0, origDescription.Format);
		if (pCpuReadTexture)
		{
			imcopy = new BYTE[screenSize];
			devcon->CopyResource(pCpuReadTexture, pNewTexture);
			if(cudaFail(cudaMemcpy2DFromArray(imcopy, ulLineBytes, cuArray, 0, 0, ulLineBytes, gameHeight,
				cudaMemcpyDeviceToHost)))
			{
				ret = false;			
			}
			else
			{
				ret = true;
			}
			
        }
	}
//	cudaCopyCleanup(pNewTexture, pCpuReadTexture);
	
	if(ret == false)
	{
		DBOUT("problem copying with cuda, skipping");
	}
	return ret;
}

