using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Console
{
    public class Util
    {
        public static void GetSample(string srcF, string outF)
        {
            Dictionary<string, int> nodesCnt = new Dictionary<string, int>();
            using (StreamReader sr = new StreamReader(srcF))
            {
                while (!sr.EndOfStream)
                {
                    var line = sr.ReadLine();
                    var items = line.Split(new char[] { ' ', '\t' });

                    if (items.Length == 2)
                        continue;

                    string nodeId = items[0];
                    int nodeCnt = int.Parse(items[2]);

                    if (!nodesCnt.ContainsKey(nodeId))
                    {
                        nodesCnt.Add(nodeId, 0);
                    }
                    nodesCnt[nodeId] += nodeCnt;
                }
            }

            var resultDict = nodesCnt.OrderByDescending(c => c.Value);

            List<string> nodeLis = new List<string>();
            int topNum = 100;
            int k = 0;
            foreach (var node in resultDict)
            {
                if(++k <= topNum)
                    nodeLis.Add(node.Key);
            }

            using (StreamWriter sw = new StreamWriter(outF, false, Encoding.UTF8))
            {
                var outFF = string.Format("{0}.vecs", outF);
                using (StreamWriter sw1 = new StreamWriter(outFF, false, Encoding.UTF8))
                {
                    using (StreamReader sr = new StreamReader(srcF))
                    {
                        while (!sr.EndOfStream)
                        {
                            var line = sr.ReadLine();
                            var items = line.Split(new char[] { ' ', '\t' });
                            if (items.Length == 2)
                                continue;

                            if (nodeLis.Contains(items[0]))
                            {
                                var outStr = string.Format("{0}_{1}_{2}", items[0], items[1], items[2]);
                                sw.WriteLine(outStr);

                                List<string> tmpLis = new List<string>();
                                for (int i = 3; i < items.Length; i++)
                                {
                                    tmpLis.Add(items[i]);
                                }
                                outStr = string.Join(",", tmpLis);
                                sw1.WriteLine(outStr);
                            }
                        }
                    }
                }
            }
        }
    }
}
