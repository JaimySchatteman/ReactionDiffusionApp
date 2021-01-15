using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Logic
{
    public class Cell
    {
        public float ChemicalA { get; set; } // Amount of chemical A
        public float ChemicalB { get; set; } // Amount of chemical B

        public Cell(float chemA, float chemB)
        {
            ChemicalA = chemA;
            ChemicalB = chemB;
        }
    }
}
