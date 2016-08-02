echo "shuf & construct dataset..."
#awk 'NR%2' ../data/dblp.authors.net > ../data/dblp.authors.net.v2

shuf ../data/dblp.authors.triple > ../data/dblp.authors.triple.shuf

head -3548292 ../data/dblp.authors.triple.shuf > ../data/tmp
awk '{print $0; print $2" "$1" "$3" "$4; print $3" "$1" "$2" "$4}' ../data/tmp > ../data/tmp1
shuf ../data/tmp1 > ../data/dblp.authors.triple.shuf.train

rm ../data/tmp
rm ../data/tmp1

tail -14193172 ../data/dblp.authors.triple.shuf > ../data/tmp #../data/dblp.authors.triple.shuf.test
awk '{print $1" "$2; print $1" "$3; print $2" "$3;}' ../data/tmp > ../data/dblp.authors.triple.shuf.test

rm ../data/dblp.authors.triple.shuf
rm ../data/tmp

echo "done!"

echo "filtering ..."
python filter.py ../data/dblp.authors.triple.shuf.train ../data/dblp.authors.triple.shuf.test ../data/dblp.authors.triple.shuf.test.v2

echo "embedding learning..."

g++ -lm -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result ms_line.cpp -o ms_line -lgsl -lm -lgslcblas

./ms_line -train ../data/dblp.authors.triple.shuf.train -output vec_2nd_wo_norm.txt -binary 0 -size 50 -order 2 -negative 2 -samples 1000 -threads 20 -sense -1 -gap -0.05 -ratio 0.2 -factor 0.75

echo "predict auc"
g++ predict.cpp -o predict
./predict ../data/node.lis.v2 vec_2nd_wo_norm.txt vec_2nd_wo_norm.txt.multi.context.emb ../data/dblp.authors.triple.shuf.train ../data/dblp.authors.triple.shuf.test.v2 testSet.predict misSet.predict 50 10
