using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Console
{
    class Program
    {
        static void Main(string[] args)
        {
            string srcF = @"F:\AAAI2017\NetEmb\Data\vec_2nd_wo_norm.txt";
            string outF = @"F:\AAAI2017\NetEmb\Data\sample";

            Util.GetSample(srcF, outF);

        }
    }
}
