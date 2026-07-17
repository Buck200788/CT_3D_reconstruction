#include <iostream>
#include "./include/BaseRecon.h"
#include <omp.h>
int main()
{
	std::cout << "available threads on CPU£º" << omp_get_max_threads() << std::endl;
	return 0;
}