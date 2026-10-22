#include <iostream>
#include "ready_graph.h"
int main(){for(int rotary=0;rotary<2;rotary++)for(int affine=0;affine<2;affine++)
for(int self:{96,168}){std::cout<<rotary<<" "<<affine<<" "<<self<<"\n";
for(int n=0;n<28;n++){auto s=ready_node(n,7168,128,self,rotary,affine);std::cout<<s.dependencies<<" "<<s.tiles<<"\n";}}
for(int p=0;p<28;p++)std::cout<<ready_priority(p)<<" ";}
