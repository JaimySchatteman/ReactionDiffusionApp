using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Logic
{
    public interface ICellGrid
    {
        int Width { get; set; }
        int Height { get; set; }

        Cell[,] Grid { get; set; }

        void FillGrid();
    }

}
