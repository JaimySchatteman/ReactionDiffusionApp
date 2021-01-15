using System;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace Logic
{
    // References:  https://www.karlsims.com/rd.html
    //              https://www.youtube.com/watch?v=BV9ny785UNc
    public class ReactionDiffusion : IReactionDiffusion
    {
        private ICellGrid CurrentCells;
        private ICellGrid NextCells;
        private float DiffusionRateA;
        private float DiffusionRateB;
        private float Feed;
        private float KillRate;

        // Output Image Properties
        // Reference: https://docs.microsoft.com/en-us/dotnet/api/system.windows.media.imaging.bitmapsource?view=net-5.0
        private PixelFormat BitmapPixelFormat;
        private int BytesPerPixel;
        private int Stride;
        private byte[] RawImage;

        public ReactionDiffusion()
        {
            
        }

        public ReactionDiffusion(int width, int height, float diffusionRateA, float diffusionRateB, float killRate, float feed)
        {
            CurrentCells = new CellGrid(width, height);
            NextCells = new CellGrid(width, height);
            CurrentCells.FillGrid();
            NextCells.FillGrid();

            DiffusionRateA = diffusionRateA;
            DiffusionRateB = diffusionRateB;
            KillRate = killRate;
            Feed = feed;

            SetAndCalculateImageFormatProperties();
        }

        public void SetAndCalculateImageFormatProperties()
        {
            BitmapPixelFormat = PixelFormats.Bgra32;
            BytesPerPixel = BitmapPixelFormat.BitsPerPixel / 8;
            Stride = CurrentCells.Width * BytesPerPixel;
            RawImage = new byte[Stride * CurrentCells.Height];
        }
        

        public BitmapSource Diffuse()
        {
            for (int x = 0; x < CurrentCells.Width; x++)
            {
                for (int y = 0; y < CurrentCells.Height; y++)
                {
                    if(x != 0 && y != 0 && x != CurrentCells.Width - 1 && y != CurrentCells.Height - 1)
                        NextCells.Grid[x, y] = DiffuseCell(CurrentCells.Grid[x, y], x, y);

                    SetColor(x, y, NextCells.Grid[x, y]);
                }
            }

            CurrentCells = NextCells;

            return BitmapSource.Create(CurrentCells.Width, CurrentCells.Height, 96, 96, BitmapPixelFormat, null, RawImage, Stride);
        }

        private Cell DiffuseCell(Cell currentCell, int x, int y)
        {
            float NextA = CalculateNewValueA(currentCell, x, y);
            float NextB = CalculateNewValueB(currentCell, x, y);

            return new Cell(NextA, NextB);
        }

        private float CalculateNewValueA(Cell currCell, int x, int y)
        {
            return currCell.ChemicalA + (DiffusionRateA * CalculateLaPlacianFunction("A", x, y) - currCell.ChemicalA * currCell.ChemicalB * currCell.ChemicalB + Feed * (1 - currCell.ChemicalA));
        }

       private float CalculateNewValueB(Cell currCell, int x, int y)
        {
            return currCell.ChemicalB + (DiffusionRateB * CalculateLaPlacianFunction("B", x, y) + currCell.ChemicalA * currCell.ChemicalB * currCell.ChemicalB - (KillRate + Feed) * currCell.ChemicalB);
        }

        // Reference: https://www.programming-techniques.com/2013/02/calculating-convolution-of-image-with-c_2.html
        // Convolution matrix [ [ 0.05f, 0.2f, 0.05f ], 
        //                      [ 0.2f,  -1,   0.2f  ], 
        //                      [ 0.05f, 0.2f, 0.05f ] ]
        public float CalculateLaPlacianFunction(string chemical, int x, int y)
        {
            float convolutionSum = 0; 

            for (int j = -1; j < 2; j++)
            {
                for(int i = -1; i < 2; i++)
                {
                    if (j != 0 && i != 0)
                    {
                        if (chemical == "A")
                            convolutionSum += CurrentCells.Grid[x + i, y + j].ChemicalA * 0.05f;
                        else
                            convolutionSum += CurrentCells.Grid[x + i, y + j].ChemicalB * 0.05f;

                    }
                    else if (j == 0 && i == 0)
                    {
                        if (chemical == "A")
                            convolutionSum += CurrentCells.Grid[x + i, y + j].ChemicalA * -1;
                        else
                            convolutionSum += CurrentCells.Grid[x + i, y + j].ChemicalB * -1;

                    }
                    else
                    {
                        if (chemical == "A")
                            convolutionSum += CurrentCells.Grid[x + i, y + j].ChemicalA * 0.2f;
                        else
                            convolutionSum += CurrentCells.Grid[x + i, y + j].ChemicalB * 0.2f;

                    }
                }
            }
            
            return convolutionSum;
        }
        // Reference: https://docs.microsoft.com/en-us/dotnet/api/system.windows.media.imaging.writeablebitmap?view=net-5.0
        private void SetColor(int x, int y, Cell cell)
        {
            RawImage[x * 4 + y * CurrentCells.Width * 4] = (byte) (255 * cell.ChemicalB);          // Blue
            RawImage[x * 4 + y * CurrentCells.Width * 4 + 1] = 0;                                 // Green
            RawImage[x * 4 + y * CurrentCells.Width * 4 + 2] = (byte) (255 * cell.ChemicalA);    // Red
            RawImage[x * 4 + y * CurrentCells.Width * 4 + 3] = 255;
        }
    }
}
