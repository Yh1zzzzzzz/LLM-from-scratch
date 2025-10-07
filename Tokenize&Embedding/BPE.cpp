#include <iostream>
#include <map>
using namespace std;
map<int , int> BPE(vector<int> Vec, int size)
{
    map<pair<int, int> , int> Vocab;
    while(Vocab.size() < size)
    {   
         //step1 : 判断是否已经在 Vocabulary中，对其进行替换
        for(int i = 0; i < Vec.size() - 1; i++)
        {
            std::pair Window = std::make_pair(Vec[i], Vec[i  + 1]);
            if(Vocab.find(Window) != Vocab.end())
            {
                Vec[i] = Vocab[Window];
                Vec.erase(Vec.begin() + i + 1);
            }
        }
        map<pair<int, int> , int> Cache; 
        for(int i = 0; i < Vec.size() - 1; i++)
        {
            std::pair Window = std::make_pair(Vec[i], Vec[i  + 1]);
            Cache[Window]++;
        }
        //扫表，建立个数
        
        int size = Cache.size();
        int max = 0;
        pair<int, int> most_common_pair;
        for(auto key : Cache)
        {
            if(key.second > max)
            {
                max = key.second;
                most_common_pair = key.first;
            }
        }
        Vocab.insert(make_pair(most_common_pair, Vocab.size()+1));
        //插入最多出现的两个token对
    }
}
int main()
{
    return 0;
}