using Cloo;
using OpenTK.Graphics;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace Logic
{
    // References:  https://www.karlsims.com/rd.html
    //              https://www.youtube.com/watch?v=BV9ny785UNc
    public class ReactionDiffusionOpenCL : IReactionDiffusion
    {
        // Canvas fields
        private int Width;
        private int Height;
        private float[] CurrImageVector;
        private ComputeImage2D CurrImage;
        private float[] NextImageVector ;
        private ComputeImage2D NextImage ;

        // Diffusion fields
        private ComputeBuffer<float> DiffusionRateA ;
        private ComputeBuffer<float> DiffusionRateB;
        private ComputeBuffer<float> Feed;
        private ComputeBuffer<float> KillRate;

        private readonly float[] ConvolutionMatrix = new float[] { 0.05f, 0.2f, 0.05f, 0.2f, -1, 0.2f, 0.05f, 0.2f, 0.05f };
        private ComputeBuffer<float> ConvolutionMatrixCL;
        private ComputeBuffer<int> ConvolutionMatrixSizeCL;

        // Compute context fields
        private ComputeContext Ctx;
        private ComputeKernel DiffuseKernel;
        private ComputeKernel InitilizeGridKernel;
        private ComputeCommandQueue CQ;
        private bool WithLocalMem;

        private readonly string InitilizeGridKernelString = @"
                     __kernel void InitializeGrid(  __read_only  image2d_t currentImage,
                                                    __write_only image2d_t nextImage)
                    {
                        const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;                        
                            
                        int2 coordinate = (int2)(get_global_id(0), get_global_id(1));
                        float4 cell = read_imagef(currentImage, smp, coordinate);    

                        int width = get_image_width(currentImage);
                        int height = get_image_height(currentImage);
                        if(coordinate.x > (width >> 1) - (width * 0.2f) && coordinate.x < (width >> 1) + (width * 0.2f) && coordinate.y > (height >> 1) - (height * 0.2f) && coordinate.y < (height >> 1) + (height * 0.2f))
                        {
                            cell.x = 0.0f;
                            cell.y = 0.0f;
                            cell.z = 1.0f;
                            cell.w = 1.0f;
                            write_imagef(nextImage, coordinate, cell);  
                        }   
                        else 
                        {
                            cell.x = 1.0f;
                            cell.y = 0.0f;
                            cell.z = 0.0f;
                            cell.w = 1.0f;
                            write_imagef(nextImage, coordinate, cell);  
                        }
                    }";

        // Reference: http://www.cmsoft.com.br/opencl-tutorial/case-study-high-performance-convolution-using-opencl-__local-memory/
        private readonly string DiffuseKernelWithLocalMemString = @"
                      __kernel void DiffuseKernelWithLocalMem(   __read_only    image2d_t currentImage,
                                                    __write_only  image2d_t nextImage,                                                  
                                                    __constant    float * diffusionRateA,
                                                    __constant    float * diffusionRateB,
                                                    __constant    float * feed,
                                                    __constant    float * killRate,
                                                    __constant    float * convolutionMatrix,
                                                    __constant    int * convuloutionMatrixWidth)
                    {
                        const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;   
                        
                        int groupIdX = get_group_id(0);
                        int groupIdY = get_group_id(1); 

                        int localIdX = get_local_id(0);
                        int localIdY = get_local_id(1); 

                        int localMemoryMatrixIdX = localIdX + 1;
                        int localMemoryMatrixIdY = localIdY + 1;

                        int localGroupSizeX = get_local_size(0);
                        int localGroupSizeY = get_local_size(1);

                        int globalIdX = groupIdX * localGroupSizeX + localIdX;
                        int globalIdY = groupIdY * localGroupSizeY + localIdY;
                        int2 globalCoordinates = (int2)(globalIdX, globalIdY);

                        int convWidthDivByTwoFloored = convuloutionMatrixWidth[0] >> 1;

                        __local float4 P[30][10];                                                         
                        P[localMemoryMatrixIdX][localMemoryMatrixIdY] = read_imagef(currentImage, smp, globalCoordinates);                    

                        if (localIdX == 0) {
                            globalCoordinates.x = globalIdX - 1;
                            globalCoordinates.y = globalIdY;
                            P[0][localMemoryMatrixIdY] = read_imagef(currentImage, smp, globalCoordinates);                          
                                                
                        } else if (localIdX == localGroupSizeX - 1) {
                            globalCoordinates.x = globalIdX + 1;
                            globalCoordinates.y = globalIdY;
                            P[localGroupSizeX + 1][localMemoryMatrixIdY] = read_imagef(currentImage, smp, globalCoordinates);                                                
                        } 


                        if (localIdY == 0) {
                            globalCoordinates.x = globalIdX;
                            globalCoordinates.y = globalIdY - 1;
                            P[localMemoryMatrixIdX][0] = read_imagef(currentImage, smp, globalCoordinates);   

                        } else if (localIdY == localGroupSizeY - 1) {
                            globalCoordinates.x = globalIdX;
                            globalCoordinates.y = globalIdY + 1;
                            P[localMemoryMatrixIdX][localGroupSizeY + 1] = read_imagef(currentImage, smp, globalCoordinates);
                        }


                       if (localIdX == 0 && localIdY == 0) {
                            globalCoordinates.x = globalIdX - 1;
                            globalCoordinates.y = globalIdY - 1;
                            P[0][0] = read_imagef(currentImage, smp, globalCoordinates);  

                        } else if (localIdX == 0 && localIdY == localGroupSizeY - 1) {
                            globalCoordinates.x = globalIdX - 1;
                            globalCoordinates.y = globalIdY + 1;
                            P[0][localGroupSizeY + 1] = read_imagef(currentImage, smp, globalCoordinates);  

                        } else if (localIdX == localGroupSizeX - 1 && localIdY == 0) {
                            globalCoordinates.x = globalIdX + 1;
                            globalCoordinates.y = globalIdY - 1;
                            P[localGroupSizeX + 1][0] = read_imagef(currentImage, smp, globalCoordinates);  

                        } else if (localIdX == localGroupSizeX - 1 && localIdY == localGroupSizeY - 1) {
                            globalCoordinates.x = globalIdX + 1;
                            globalCoordinates.y = globalIdY + 1;
                            P[localGroupSizeX + 1][localGroupSizeY + 1] = read_imagef(currentImage, smp, globalCoordinates);                                                
                        }                        

                        barrier(CLK_LOCAL_MEM_FENCE);
                        
                        float convolutionSumA = 0;
                        float convolutionSumB = 0;                      
                        #pragma unroll
                        for (int j = -convWidthDivByTwoFloored; j <= convWidthDivByTwoFloored; j++)
                        {
                            #pragma unroll
                            for(int i = -convWidthDivByTwoFloored; i <= convWidthDivByTwoFloored; i++)
                            {                                
                                float4 convolutionCell = P[localMemoryMatrixIdX + i][localMemoryMatrixIdY + j];                              
                              
                                convolutionSumA += convolutionCell.x * convolutionMatrix[(i + convWidthDivByTwoFloored)  + ((j + convWidthDivByTwoFloored) * convuloutionMatrixWidth[0])];    
                                convolutionSumB += convolutionCell.z * convolutionMatrix[(i + convWidthDivByTwoFloored)  + ((j + convWidthDivByTwoFloored) * convuloutionMatrixWidth[0])];               
                            }
                        }                           
                           
                        barrier(CLK_LOCAL_MEM_FENCE);
                      
                        float nextValueA = P[localMemoryMatrixIdX][localMemoryMatrixIdY].x + (diffusionRateA[0] * convolutionSumA - P[localMemoryMatrixIdX][localMemoryMatrixIdY].x * P[localMemoryMatrixIdX][localMemoryMatrixIdY].z * P[localMemoryMatrixIdX][localMemoryMatrixIdY].z + feed[0] * (1 - P[localMemoryMatrixIdX][localMemoryMatrixIdY].x));
                        float nextValueB = P[localMemoryMatrixIdX][localMemoryMatrixIdY].z + (diffusionRateB[0] * convolutionSumB + P[localMemoryMatrixIdX][localMemoryMatrixIdY].x * P[localMemoryMatrixIdX][localMemoryMatrixIdY].z * P[localMemoryMatrixIdX][localMemoryMatrixIdY].z - (killRate[0] + feed[0]) * P[localMemoryMatrixIdX][localMemoryMatrixIdY].z);               

                        P[localMemoryMatrixIdX][localMemoryMatrixIdY].x = nextValueA;        // R  -> Chemical A
                        P[localMemoryMatrixIdX][localMemoryMatrixIdY].z = nextValueB;        // B  -> Chemical B
                     
                        write_imagef(nextImage, (int2)(globalIdX, globalIdY),  P[localMemoryMatrixIdX][localMemoryMatrixIdY]);                        
                      
                    }";

        private readonly string DiffuseKernelString = @"
                     __kernel void DiffuseKernel(   __read_only   image2d_t currentImage,
                                                    __write_only  image2d_t nextImage,                                                  
                                                    __constant    float * diffusionRateA,
                                                    __constant    float * diffusionRateB,
                                                    __constant    float * feed,
                                                    __constant    float * killRate,
                                                    __constant    float * convolutionMatrix,
                                                    __constant    int * convuloutionMatrixWidth)
                    {
                        const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;   

                        float convolutionSumA = 0;
                        float convolutionSumB = 0;
                        int convWidthDivByTwo = convuloutionMatrixWidth[0] >> 1;

                        #pragma unroll
                        for (int j = -convWidthDivByTwo; j <= convWidthDivByTwo; j++)
                        {
                            #pragma unroll
                            for(int i = -convWidthDivByTwo; i <= convWidthDivByTwo; i++)
                            {
                                int2 convolutionCoor = (int2)(get_global_id(0) + i, get_global_id(1) + j);
                                float4 convolutionCell = read_imagef(currentImage, smp, convolutionCoor);                              
                              
                                convolutionSumA += convolutionCell.x * convolutionMatrix[(i + convWidthDivByTwo) + (j + convWidthDivByTwo) * convuloutionMatrixWidth[0]];    
                                convolutionSumB += convolutionCell.z * convolutionMatrix[(i + convWidthDivByTwo) + (j + convWidthDivByTwo) * convuloutionMatrixWidth[0]];               
                            }
                        }

                        int2 coordinate = (int2)(get_global_id(0), get_global_id(1));
                        float4 cell = read_imagef(currentImage, smp, coordinate);                     

                        float nextValueA = cell.x + (diffusionRateA[0] * convolutionSumA - cell.x * cell.z * cell.z + feed[0] * (1 - cell.x));
                        float nextValueB = cell.z + (diffusionRateB[0] * convolutionSumB + cell.x * cell.z * cell.z - (killRate[0] + feed[0]) * cell.z);            
                       
                        cell.x = nextValueA;        // R  -> Chemical A
                        cell.z = nextValueB;        // B  -> Chemical B
                        
                        write_imagef(nextImage, coordinate, cell); 
                    }";


        public ReactionDiffusionOpenCL()
        {       
        }

        public ReactionDiffusionOpenCL(int width, int height, float diffusionRateA, float diffusionRateB, float feed, float killRate, bool withLocalMem)
        {
            Width = width;
            Height = height;
            WithLocalMem = withLocalMem;
            SetupContext();
            InitializeDiffusionKernelFields(diffusionRateA, diffusionRateB, feed, killRate);
            InitializeCellGrid();
            SetupDiffuseKernel();
        }

        private void SetupContext()
        {
            ComputeContextPropertyList Properties = new ComputeContextPropertyList(ComputePlatform.Platforms[1]); 
            Ctx = new ComputeContext(ComputeDeviceTypes.All, Properties, null, IntPtr.Zero);         
        }

        // Reference: http://www.cmsoft.com.br/opencl-tutorial/opencl-image2d-variables/
        private void InitializeDiffusionKernelFields(float diffusionRateA, float diffusionRateB, float feed, float killRate)
        {
            InitializeImageFieds();
            InitializeDiffusionFields( diffusionRateA,  diffusionRateB, feed,  killRate);
        }

        private void InitializeImageFieds()
        {
            int ImageVectorSizeInFloats = Width * Height * 4;
            CurrImageVector = new float[ImageVectorSizeInFloats];
            NextImageVector = new float[ImageVectorSizeInFloats];

            unsafe
            {
                fixed (float* imgPtr = CurrImageVector)
                {
                    ComputeImageFormat format = new ComputeImageFormat(ComputeImageChannelOrder.Rgba, ComputeImageChannelType.Float);
                    CurrImage = new ComputeImage2D(Ctx, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, format, Width, Height, Width * 4 * sizeof(float), (IntPtr)imgPtr);
                    NextImage = new ComputeImage2D(Ctx, ComputeMemoryFlags.ReadWrite, format, Width, Height, 0, IntPtr.Zero);
                }
            }
        }

        private void InitializeDiffusionFields(float diffusionRateA, float diffusionRateB, float feed, float killRate)
        {
            DiffusionRateA = new ComputeBuffer<float>(Ctx, ComputeMemoryFlags.CopyHostPointer | ComputeMemoryFlags.ReadWrite, new float[] { diffusionRateA });
            DiffusionRateB = new ComputeBuffer<float>(Ctx, ComputeMemoryFlags.CopyHostPointer | ComputeMemoryFlags.ReadWrite, new float[] { diffusionRateB });
            Feed = new ComputeBuffer<float>(Ctx, ComputeMemoryFlags.CopyHostPointer | ComputeMemoryFlags.ReadWrite, new float[] { feed });
            KillRate = new ComputeBuffer<float>(Ctx, ComputeMemoryFlags.CopyHostPointer | ComputeMemoryFlags.ReadWrite, new float[] { killRate });

            ConvolutionMatrixCL = new ComputeBuffer<float>(Ctx, ComputeMemoryFlags.CopyHostPointer | ComputeMemoryFlags.ReadWrite, ConvolutionMatrix);
            ConvolutionMatrixSizeCL = new ComputeBuffer<int>(Ctx, ComputeMemoryFlags.CopyHostPointer | ComputeMemoryFlags.ReadWrite, new int[] { (int)Math.Sqrt(ConvolutionMatrix.Length) });

        }

        private void InitializeCellGrid()
        {
            ComputeProgram prog = new ComputeProgram(Ctx, InitilizeGridKernelString);
            prog.Build(Ctx.Devices, "", null, IntPtr.Zero);
            InitilizeGridKernel = prog.CreateKernel("InitializeGrid");
            InitilizeGridKernel.SetMemoryArgument(0, CurrImage);
            InitilizeGridKernel.SetMemoryArgument(1, NextImage);
            CQ = new ComputeCommandQueue(Ctx, Ctx.Devices[0], ComputeCommandQueueFlags.None);
            CQ.Execute(InitilizeGridKernel, null, new long[] { Width, Height }, null, null);
        }

        private void SetupDiffuseKernel()
        {
            ComputeProgram prog;
            if (WithLocalMem)
            {
                prog = new ComputeProgram(Ctx, DiffuseKernelWithLocalMemString);
                prog.Build(Ctx.Devices, "", null, IntPtr.Zero);
                DiffuseKernel = prog.CreateKernel("DiffuseKernelWithLocalMem");
            }
            else
            {
                prog = new ComputeProgram(Ctx, DiffuseKernelString);
                prog.Build(Ctx.Devices, "", null, IntPtr.Zero);
                DiffuseKernel = prog.CreateKernel("DiffuseKernel");
            }
            
            SetMemoryArguments();
        }

        private void SetMemoryArguments()
        {
            DiffuseKernel.SetMemoryArgument(0, CurrImage);
            DiffuseKernel.SetMemoryArgument(1, NextImage);
            DiffuseKernel.SetMemoryArgument(2, DiffusionRateA);
            DiffuseKernel.SetMemoryArgument(3, DiffusionRateB);
            DiffuseKernel.SetMemoryArgument(4, Feed);
            DiffuseKernel.SetMemoryArgument(5, KillRate);
            DiffuseKernel.SetMemoryArgument(6, ConvolutionMatrixCL);
            DiffuseKernel.SetMemoryArgument(7, ConvolutionMatrixSizeCL);
        }

        public BitmapSource Diffuse()
        {
            CurrImage = NextImage;
            SetMemoryArguments();

            if (WithLocalMem)
                CQ.Execute(DiffuseKernel, null, new long[] { Width, Height }, new long[] { 28, 8 }, null);
            else
                CQ.Execute(DiffuseKernel, null, new long[] { Width, Height }, null, null);          
            
            return ReadImageDataFromGPUMemory();
        }

        private BitmapSource ReadImageDataFromGPUMemory()
        {
            unsafe
            {
                fixed (float* imgPtr = NextImageVector)
                {
                    CQ.Read(NextImage, true, new SysIntX3(0, 0, 0), new SysIntX3(Width, Height, 1), Width * 4 * sizeof(float), 0, (IntPtr)imgPtr, null);
                }
            }

            return BitmapSource.Create(Width, Height, 96, 96, PixelFormats.Rgba128Float, null, NextImageVector, (PixelFormats.Rgba128Float.BitsPerPixel / 8) * Width);
        }
    }
}
