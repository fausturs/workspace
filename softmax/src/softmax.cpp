#include <iostream>
#include <valarray>
#include <vector>
#include <random>

#include "../../tools/read_mnist.hpp"
#include "../../tools/Timer.hpp"

class Softmax
{
	//求解的目标函数形式为:
	/*
	J(\theta) = -\frac{1}{m}
	\left[
		\sum_{i=1}^m \sum_{j=1}^k
			1\{y^{(i)}=j\}
			\log\frac{e^{\theta^T_j x^{(i)}}}{\sum_{l=1}^k e^{\theta^T_l x^{(i)}}}
	\right]
	+ \frac{\lambda}{2}\sum_{i=1}^k \sum_{j=0}^n \theta_{ij}^2
	*/
	typedef std::valarray< double > parameter_vector_type;
	typedef std::valarray< std::valarray<double> > parameter_matrix_type;
	parameter_matrix_type theta;

	//在theta作为参数的情况下，样本x被分为y类的概率
	template<class T>
	double probability(const T& x, size_t y)
	{
		double deno = 0.0;
		//计算分母
		for (auto& theta_i: theta)
			deno += std::exp(std::inner_product(std::begin(theta_i), std::end(theta_i), std::begin(x), 0.0));
		//计算分子
		double nume = std::inner_product(std::begin(theta[y]), std::end(theta[y]), std::begin(x), 0.0);
		nume = std::exp(nume);
		return nume/deno;
	}
public:
	size_t feature_num;
	size_t label_num;
	double init_learning_rate;
	double norm_weight;
	Softmax(size_t label_num, double norm_weight = 0.0, double init_learning_rate = 0.5):theta{},feature_num{0}
	{
		this->label_num = label_num;
		this->init_learning_rate = init_learning_rate;
		this->norm_weight = norm_weight;
	}

	Softmax(const Softmax&) = delete;
	Softmax(Softmax&& rhs)
	{
		*this = std::move(rhs);
	}
	Softmax& operator=(const Softmax&) = delete;
	Softmax& operator=(Softmax&& rhs)
	{
		theta = std::move(rhs.theta);
		feature_num = rhs.feature_num;
		label_num = rhs.label_num;
		init_learning_rate = rhs.init_learning_rate;
		norm_weight = rhs.norm_weight;
		return *this;
	}

	template<class Data_tp, class Label_tp>
	void train(const Data_tp& data, const Label_tp& labels, size_t epoch_num = 1000, size_t iter_num = 1)
	{
		if (labels.size() == 0) return;
		feature_num = data[0].size();
		//随机生成theta矩阵
		theta.resize(label_num);
		for (auto& theta_i : theta)
		{
			theta_i.resize(feature_num);
			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_real_distribution<double> dis(-1, 1);
			for (auto& theta_ij : theta_i) theta_ij = dis(gen); 
		}
		//梯度下降
		parameter_vector_type gradient;	
		//一些输出的参数
		std::cout<<"epoch\t\ttime_ms\t\tloss\n";
		wjy::Timer timer;
		timer.start();
		for (size_t epoch=0;epoch<epoch_num;epoch++)
		{
			timer.end();
			std::cout<<epoch<<"\t\t"<<timer.get_duration_ms()<<"\t\t"<<get_loss(data, labels)<<"\n";
			//每个epoch迭代次数是iter_num，默认是1
			for (size_t iter=0;iter<iter_num;iter++)
			{
				for (size_t j=0;j<theta.size();j++)
				{
					//计算梯度
					gradient = theta[j]*norm_weight;
					double temp;
					for (int i=0;i<data.size();i++)
					{
						if (labels[i] == j) temp = 1;
						temp -= probability(data[i], j); 
						temp /= (double)-labels.size();
						gradient += data[i]*temp;  
					}
					//梯度下降
					theta[j] -= gradient * (init_learning_rate)/(1+epoch);
				}
			}
		}

	}
	
	template<class Data_tp, class Label_tp>
	void test(const Data_tp& data, const Label_tp& labels )
	{
		size_t cnt=0, n=data.size();
		for (size_t i=0;i<n;i++)
		{
			double p = probability(data[i], labels[i]);
			for (size_t l = 0;l<label_num;l++)
			{
				if (l!=labels[i] && probability(data[i], l)>p)
				{
					cnt++;
					break;
				}
			}
		}
		std::cout<<"正确率："<<(n-cnt)/(double)(n)<<"\n";
	}

	template<class Data_tp, class Label_tp>
	double get_loss(const Data_tp& data, const Label_tp& labels)
	{
		//计算正则化项
		double loss = 0.0;
		for (auto& theta_i : theta)
		{
			parameter_vector_type theta_i_2 = std::pow(theta_i, 2.0);
			loss += std::accumulate(std::begin(theta_i_2), std::end(theta_i_2), 0.0);
		}
		loss = loss * norm_weight / 2.0;
		//计算损失项
		for (int i=0;i<labels.size();i++)
		{
			//累计loss
			loss -= std::log(probability(data[i], labels[i]))/ (double)labels.size();
		}
		return loss;
	}

	~Softmax() = default;
};


int main()
{
	auto test_data = read_mnist_image_file< std::valarray< std::valarray<double> > >("../../data/mnist_data/t10k-images-idx3-ubyte", 1, 1);
	auto test_labels = read_mnist_label_file< std::valarray<int> >("../../data/mnist_data/t10k-labels-idx1-ubyte");
	
	auto train_data = read_mnist_image_file< std::valarray< std::valarray<double> > >("../../data/mnist_data/train-images-idx3-ubyte", 1, 1);
	auto train_labels = read_mnist_label_file< std::valarray<int> >("../../data/mnist_data/train-labels-idx1-ubyte");

	//归一化
	for (auto & x : train_data)
		for (auto & y : x) y = y/255.0;
	for (auto & x : test_data)
		for (auto & y : x) y = y/255.0;

	Softmax sm(10, 1);
	std::cout<<"训练数据集大小:"<<train_data.size()<<"\n";
	sm.train(train_data, train_labels, 10);

	std::cout<<"训练数据集上的";
	sm.test(train_data, train_labels);
	std::cout<<"测试数据集上的";
	sm.test(test_data, test_labels);
	return 0;
}
