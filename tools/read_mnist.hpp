#ifndef READ_MNIST_HPP
#define READ_MNIST_HPP

#include <iostream>
#include <fstream>

#include <string>

//将一个整数类型T类型的变量x    按照字节为单位进行前后翻转
//x的位数有bits_num来指定
template<class T>
T reverse_by_byte(T x, int bits_num = 32)
{
	T temp = 0xff;
	T ans = 0;
	for (int i=0;i<bits_num/8;i++)
	{
		ans = (ans << 8) + (temp & x);
		x = x >> 8;
	}
	return ans;
}

//T类型应该是一个支持resize和中括号访问符的类型
//比较理想的是valarry array 或者 vector
template<class T>
T read_mnist_label_file(std::string file_name)
{
	T labels;
	//开始处理文件
	std::ifstream myin(file_name, std::ios::binary);
	if (!myin.is_open()) return labels;
	//读取文件头
	uint32_t magic_number, label_num;
	myin.read((char*)&magic_number, sizeof(magic_number));
	myin.read((char*)&label_num, sizeof(label_num));
	//这里读入是按字节读入的 保存数据有大小端的区别
	//所以还要按字节翻转一次
	magic_number = reverse_by_byte(magic_number, 32);
	label_num = reverse_by_byte(label_num, 32);
	
	labels.resize(label_num);
	uint8_t label;
	//将所有label读入并保存
	for (int i=0;i<label_num;i++)
	{
		myin.read((char*)&label, sizeof(label));
		labels[i] = label;
	}
	return labels;
}

template<class T, class Filler_T = int> 
T read_mnist_image_file(std::string file_name, int extra_space = 0, Filler_T filler = 0)
{
	T images;
	std::ifstream myin(file_name, std::ios::binary);
	if (!myin.is_open()) return images;
	//
	uint32_t magic_number, image_num, rows, cols;
	myin.read((char*)&magic_number, sizeof(magic_number));
	myin.read((char*)&image_num, sizeof(image_num));
	myin.read((char*)&rows, sizeof(rows));
	myin.read((char*)&cols, sizeof(cols));
	//这里读入是按字节读入的 保存数据有大小端的区别
	//所以还要按字节翻转一次
	magic_number = reverse_by_byte(magic_number);
	image_num = reverse_by_byte(image_num);
	rows = reverse_by_byte(rows);
	cols = reverse_by_byte(cols);
	
	images.resize(image_num);
	uint8_t pixel;
	uint32_t pixel_num = rows*cols;
	for (int i=0;i<image_num;i++)
	{
		images[i].resize(pixel_num + extra_space);
		//按行组织的保存方式
		for (int j=0;j<pixel_num;j++)
		{
			myin.read((char*)&pixel, sizeof(pixel));
			images[i][j] = pixel;
		}
		//填充额外的位
		for(int j=0;j<extra_space;j++) images[i][j+pixel_num] = filler;
	}
	return images;
}
#endif
