using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Logic
{
    public class CellGrid : ICellGrid
    {
        public int Width { get; set; }
        public int Height { get; set; }

        public Cell[,] Grid { get; set; }

        public CellGrid()
        {

        }

        public CellGrid(int width, int height)
        {
            Width = width;
            Height = height;
            Grid = new Cell[Width, Height];
        }

        public void FillGrid()
        {
            Grid = new Cell[Width, Height];

            for (int y = 0; y < Height; y++)
            {
                for (int x = 0; x < Width; x++)
                {
                    Grid[x, y] = new Cell(1.0f, 0.0f);
                }
            }

            for (int y = (int)((Height / 2) - (Height * 0.2)); y < (int)((Height / 2) + (Height * 0.2)); y++)
            {
                for (int x = (int)((Width / 2) - (Width * 0.2)); x < (int)((Width / 2) + (Width * 0.2)); x++)
                {
                    Grid[x, y] = new Cell(0.0f, 1.0f);
                }
            }
        }
    }
}
