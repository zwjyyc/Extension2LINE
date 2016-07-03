#echo "shuf & construct dataset..."
#shuf ../data/dblp.authors.net > ../data/dblp.authors.net.shuf
#head -7304290 ../data/dblp.authors.net.shuf > ../data/dblp.authors.net.shuf.train
#tail -811588 ../data/dblp.authors.net.shuf > ../data/dblp.authors.net.shuf.test
#rm ../data/dblp.authors.net.shuf
#echo "done!"

#echo "embedding learning..."
#./../linux/line -train ../data/dblp.authors.net.shuf.train -output ../linux/vec_2nd_wo_norm.txt -binary 0 -size 50 -order 2 -negative 5 -samples 500 -threads 5
#echo "done!"

#echo "calculate auc"
#./predict ../data/node.lis ../linux/vec_2nd_wo_norm.txt ../linux/vec_2nd_wo_norm.txt.cemb ../data/dblp.authors.net.shuf.train ../data/dblp.authors.net.shuf.test testSet.predict misSet.predict
#echo "done!"

#echo "embedding learning..."

g++ -lm -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result ms_line.cpp -o ms_line -lgsl -lm -lgslcblas

./ms_line -train ../data/dblp.authors.net.shuf.train -output vec_2nd_wo_norm.txt -binary 0 -size 50 -order 2 -negative 5 -samples 1000 -threads 5 -sense -1 -gap -1 -ratio 0.5

echo "predict auc"
g++ predict.cpp -o predict
./predict ../data/node.lis vec_2nd_wo_norm.txt vec_2nd_wo_norm.txt.multi.context.emb ../data/dblp.authors.net.shuf.train ../data/dblp.authors.net.shuf.test testSet.predict misSet.predict 50

