#include<map>
#include<vector>
#include<iostream>
#include<fstream>
#include<algorithm>
#include<cmath>
#include<cstring>
#include<queue>

using namespace std;

map<string, int> name_map;
string name[2000010];
map<pair<int, int>, vector<float> > vec_map;
priority_queue<pair<pair<float, int>, int> > q;
bool v[2000010];

int main(int argc, char **argv){
	ifstream f1;
	int num;
	string st;
	f1.open(argv[1]);
	cout<<argv[1]<<endl;
	while (f1>>num){
		f1.get();
		getline(f1, st);
		int l = st.length() - 1;
		while (st[l] == '\n' || st[l] == '\r'){
			st.erase(l);
			l--;
		}
		if (name_map.find(st) == name_map.end()) name_map[st] = num;
		name[num] = st;
	}
	cout<<"finish"<<endl;
	ifstream f3(argv[2]);
	//ofstream f4("max_sim.table");
	//ofstream f5("gotten.table");
	int vec_num, dim;
	f3>>vec_num>>dim;
	//cout<<vec_num<<' '<<dim<<endl;
	int vec_id, vec_sense, vec_count;
	int num_count = 0;
	while (f3>>vec_id>>vec_sense>>vec_count){
		vector<float> vec;
		for (int j = 0; j < dim; j++){
			float x;
			f3>>x;
			vec.push_back(x);
		}
		for (int d = 0; d < dim; d++){
			vec_map[make_pair(vec_id, vec_sense)].push_back(vec[d]);
		}
		num_count++;
		if (num_count % 100000 == 0) cout<<num_count<<endl;
	}
	cout<<"done"<<endl;
	while (getline(cin, st)){
		int l = st.length() - 1;
		while (st[l] == '\n' || st[l] == '\r'){
			st.erase(l);
			l--;
		}
		num = name_map[st];
		int sense_num = 0;
		while (vec_map.find(make_pair(num, sense_num)) != vec_map.end()) sense_num++;
		cout<<"sense_num "<<sense_num<<endl;
		for (int s = 0; s < sense_num; s++){
			while(!q.empty()) q.pop();
			map<pair<int, int>, vector<float> >::iterator i = vec_map.begin();
			int count = 0;
			while (i != vec_map.end()){
				int tmp_vec_id = i->first.first;
				int tmp_vec_sense = i->first.second;
				//f5<<tmp_vec_id<<' '<<name[tmp_vec_id]<<endl;
				vector<float> tmp_vec(i->second);
				//for (int d = 0; d < dim; d++){
				//	f5<<tmp_vec[d]<<' ';
				//}
				//f5<<endl;
				if (num == tmp_vec_id){
					i++;
					continue;
				}
				float sum = 0, suma = 0, sumb = 0;
				vector<float> main_vec(vec_map[make_pair(num, s)]);
				for (int d = 0; d < dim; d++){
					suma += pow(main_vec[d], 2.0);
					sumb += pow(tmp_vec[d], 2.0);
					sum += main_vec[d] * tmp_vec[d];
				}
				suma = sqrt(suma);
				sumb = sqrt(sumb);
				if (suma != 0 && sumb != 0){
					sum = sum / (suma * sumb);
					q.push(make_pair(make_pair(sum, tmp_vec_id), tmp_vec_sense));
				}
				//if (max_sim == -10000) cout<<tmp_vec_id<<endl;
				//f4<<max_sim<<' '<<tmp_vec_id<<endl;
				i++;
				if (count % 100000 == 0) cout<<count<<endl;
				count++;
			}
			cout<<"finish counting"<<endl;
			memset(v, false, sizeof(v));
			for (int k = 0; k < 10; k++){
				if (q.empty()) break;
				if (v[q.top().first.second]){
					k--;
					q.pop();
					continue;
				}
				v[q.top().first.second] = true;
				cout<<name[q.top().first.second]<<' '<<q.top().first.first<<' '<<q.top().second<<endl;
				//for (int d = 0; d < dim; d++){
				//	cout<<vec_map[make_pair(j->second, 0)][d]<<' ';
				//}
				//cout<<endl;
				q.pop();
			}
		}
		cout<<"program finish"<<endl;
	}
}
