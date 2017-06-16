#include <iostream>
#include <valarray>

#include "../../tools/read_mnist.hpp"

int main()
{
	std::valarray<int> t = read_mnist_label_file< std::valarray<int> >("../../mnist_data/t10k-labels-idx1-ubyte");
	std::cout<< t.size() <<std::endl;
	typedef std::valarray< std::valarray<int> > data_type;
	data_type a = read_mnist_image_file< data_type >("../../mnist_data/t10k-images-idx3-ubyte", 1, 1);
	std::cout<<a.size()<<" "<<a[0].size()<<"\n";
	return 0;
}
