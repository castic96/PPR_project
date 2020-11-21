#include "../util/utils.h"
#include "network_gpu.h"


void Convert_Vectors_To_Buff(std::vector<double>& input_values, std::vector<double>& target_values, cl_float*& input_values_buff, cl_float*& target_values_buff) {
	
	for (unsigned i = 0; i < input_values.size(); i++) {
		input_values_buff[i] = (cl_float)input_values[i];
	}

	input_values.clear();
	input_values.shrink_to_fit();


	for (unsigned i = 0; i < target_values.size(); i++) {
		target_values_buff[i] = (cl_float)target_values[i];
	}

	target_values.clear();
	target_values.shrink_to_fit();
}

void Print_Device_Info(cl::Device device, unsigned i) {
	std::cout
		<< "\n   Device " << i << ": "
		<< device.getInfo<CL_DEVICE_NAME>()
		<< "\n\t Device Version:\t"
		<< device.getInfo<CL_DEVICE_VERSION >()
		<< "\n\t OpenCL C Version:\t"
		<< device.getInfo<CL_DEVICE_OPENCL_C_VERSION>()
		<< "\n\t Compute Units:\t\t"
		<< device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()
		<< "\n\t Max Work Group Size:\t"
		<< device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()
		<< "\n\t Clock Frequency:\t"
		<< device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>()
		<< "\n\t Local Memory Size:\t"
		<< device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>()
		<< "\n\t Alignment:\t\t"
		<< device.getInfo<CL_DEVICE_ADDRESS_BITS>()
		<< "\n\t Global Memory Size:\t"
		<< device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();

	std::string str = device.getInfo<CL_DEVICE_EXTENSIONS>();
	size_t found = str.find("cl_khr_fp64");
	std::cout << "\n\t Double Precision:\t";
	if (found != std::string::npos) { std::cout << "yes\n"; }
	else { std::cout << "no\n"; }

	std::cout << "\n----------------------------------------------\n";
}

void Print_Platform_Info(cl::Platform platform, unsigned i) {
	std::cout << "Platform: " << i + 1 << ": "
		<< platform.getInfo<CL_PLATFORM_NAME>()
		<< "\n----------------------------------------------"
		<< "\nVendor:\t\t" << platform.getInfo<CL_PLATFORM_VENDOR>()
		<< "\nVersion:\t" << platform.getInfo<CL_PLATFORM_VERSION>();
	std::cout << "\n----------------------------------------------\n";
}

bool Find_Devices(std::vector<cl::Device>& devices_gpu, std::vector<cl::Device>& devices_cpu) {
	std::vector<cl::Platform> available_platforms;
	std::vector<cl::Device> available_devices_gpu;
	std::vector<cl::Device> available_devices_cpu;
	cl::Device current_device;
	cl::Platform::get(&available_platforms);
	unsigned counter = 0;

	for (unsigned i = 0; i < available_platforms.size(); i++) {

		Print_Platform_Info(available_platforms[i], i);

		available_devices_gpu.clear();
		available_devices_cpu.clear();

		available_platforms[i].getDevices(CL_DEVICE_TYPE_GPU, &available_devices_gpu);
		available_platforms[i].getDevices(CL_DEVICE_TYPE_CPU, &available_devices_cpu);

		if ((available_devices_gpu.size() == 0) && (available_devices_cpu.size())) {
			continue;
		}

		for (unsigned j = 0; j < available_devices_gpu.size(); j++) {
			counter++;
			Print_Device_Info(available_devices_gpu[j], counter);
			devices_gpu.push_back(available_devices_gpu[j]);
		}

		for (unsigned j = 0; j < available_devices_cpu.size(); j++) {
			counter++;
			Print_Device_Info(available_devices_cpu[j], counter);
			devices_cpu.push_back(available_devices_cpu[j]);
		}

		counter = 0;
	}

	if (available_platforms.size() == 0 || (devices_gpu.size() == 0 && devices_cpu.size() == 0)) {
		std::cout << "Error: There are no OpenCL devices available!" << std::endl;
		return false;
	}

	return true;
}

bool Choose_Device(cl::Device*& choosen_device, std::vector<cl::Device>& devices_gpu, std::vector<cl::Device>& devices_cpu, bool prefer_gpu) {

	if (prefer_gpu) {
		if (devices_gpu.size() > 0) {
			choosen_device = new cl::Device(devices_gpu[0]);
			return true;
		}
	}

	if (devices_cpu.size() > 0) {
		choosen_device = new cl::Device(devices_cpu[0]);
		return true;
	}

	if (!prefer_gpu) {
		if (devices_gpu.size() > 0) {
			choosen_device = new cl::Device(devices_gpu[0]);
			return true;
		}
	}

	return false;
}

void Load_Code(std::string file, std::string& cl_code) {
	std::ifstream fileStream(file);
	std::stringstream buffer;
	buffer << fileStream.rdbuf();

	cl_code = buffer.str();
}

bool Build_Cl_Program(cl::Device*& device, cl::Context& context, cl::Program& program, std::string& cl_code) {
	cl::Program::Sources sources;

	sources.push_back({ cl_code.c_str(), cl_code.length() });

	context = cl::Context({ *device });
	program = cl::Program(context, sources);

	if (program.build({ *device }) != CL_SUCCESS)
	{
		std::cout << "Build error: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(*device) << "\n";
		return false;
	}
	else
	{
		std::cout << "Build OK.\n";
		return true;
	}
}

void Allocate_Buffers(kiv_ppr_network_gpu::TNetworkGPU& network, cl::Context& context, unsigned input_size, unsigned target_size) {
	network.cl_buff_neural_net_data = new cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * CL_BUFF_NEURAL_NET_DATA_SIZE);
	network.cl_buff_delta_gradient_data = new cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * CL_BUFF_DELTA_GRADIENT_DATA_SIZE);
	network.cl_buff_input_data = new cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * input_size);
	network.cl_buff_target_data = new cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * target_size);
	network.cl_buff_helper_data = new cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * HELPER_DATA_BUFF_SIZE);
	network.cl_buff_result_indexes = new cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_int) * (network.num_of_training_sets + 1));
	network.cl_buff_training_set_id = new cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_int));

	network.neural_net_buff = (cl_float*)calloc(CL_BUFF_NEURAL_NET_DATA_SIZE, sizeof(cl_float));
	network.delta_gradient_buff = (cl_float*)calloc(CL_BUFF_DELTA_GRADIENT_DATA_SIZE, sizeof(cl_float));
	network.input_values_buff = (cl_float*)calloc(input_size, sizeof(cl_float));
	network.target_values_buff = (cl_float*)calloc(target_size, sizeof(cl_float));
	network.helper_buff = (cl_float*)calloc(HELPER_DATA_BUFF_SIZE, sizeof(cl_float));
	network.result_indexes_buff = (cl_int*)calloc((network.num_of_training_sets + 1), sizeof(cl_int));
	network.training_set_id_buff = (cl_int*)calloc(1, sizeof(cl_int));
}

void Init_Network_Data(kiv_ppr_network_gpu::TNetworkGPU& network) {
	network.neural_net_buff[kiv_ppr_mapping_gpu::input_neuron_i(INPUT_LAYER_NEURONS_COUNT)] = 1.0f;
	network.neural_net_buff[kiv_ppr_mapping_gpu::hidden1_neuron_i(HIDDEN1_LAYER_NEURONS_COUNT)] = 1.0f;
	network.neural_net_buff[kiv_ppr_mapping_gpu::hidden2_neuron_i(HIDDEN2_LAYER_NEURONS_COUNT)] = 1.0f;
	network.neural_net_buff[kiv_ppr_mapping_gpu::output_neuron_i(OUTPUT_LAYER_NEURONS_COUNT)] = 1.0f;
}

void Init_Training_Set_Id(kiv_ppr_network_gpu::TNetworkGPU& network) {
	network.training_set_id_buff[0] = -1;
}

void Init_Weights(kiv_ppr_network_gpu::TNetworkGPU& network) {

	for (unsigned i = 0; i <= INPUT_LAYER_NEURONS_COUNT; i++) {
		for (unsigned j = 0; j < HIDDEN1_LAYER_NEURONS_COUNT; j++) {
			network.neural_net_buff[kiv_ppr_mapping_gpu::weight_input_hidden1(i, j)] = (cl_float)kiv_ppr_utils::Get_Random_Weight();
		}
	}

	for (unsigned i = 0; i <= HIDDEN1_LAYER_NEURONS_COUNT; i++) {
		for (unsigned j = 0; j < HIDDEN2_LAYER_NEURONS_COUNT; j++) {
			network.neural_net_buff[kiv_ppr_mapping_gpu::weight_hidden1_hidden2(i, j)] = (cl_float)kiv_ppr_utils::Get_Random_Weight();
		}
	}

	for (unsigned i = 0; i <= HIDDEN2_LAYER_NEURONS_COUNT; i++) {
		for (unsigned j = 0; j < OUTPUT_LAYER_NEURONS_COUNT; j++) {
			network.neural_net_buff[kiv_ppr_mapping_gpu::weight_hidden2_output(i, j)] = (cl_float)kiv_ppr_utils::Get_Random_Weight();
		}
	}

}

void Write_Init_Data_To_Buffers(kiv_ppr_network_gpu::TNetworkGPU& network, unsigned input_size, unsigned target_size) {
	network.queue->enqueueWriteBuffer(*(network.cl_buff_neural_net_data), CL_TRUE, 0, sizeof(cl_float) * CL_BUFF_NEURAL_NET_DATA_SIZE, network.neural_net_buff);
	network.queue->enqueueWriteBuffer(*(network.cl_buff_delta_gradient_data), CL_TRUE, 0, sizeof(cl_float) * CL_BUFF_DELTA_GRADIENT_DATA_SIZE, network.delta_gradient_buff);
	network.queue->enqueueWriteBuffer(*(network.cl_buff_input_data), CL_TRUE, 0, sizeof(cl_float) * input_size, network.input_values_buff);
	network.queue->enqueueWriteBuffer(*(network.cl_buff_target_data), CL_TRUE, 0, sizeof(cl_float) * target_size, network.target_values_buff);
	network.queue->enqueueWriteBuffer(*(network.cl_buff_helper_data), CL_TRUE, 0, sizeof(cl_float) * HELPER_DATA_BUFF_SIZE, network.helper_buff);
	network.queue->enqueueWriteBuffer(*(network.cl_buff_result_indexes), CL_TRUE, 0, sizeof(cl_int) * (network.num_of_training_sets + 1), network.result_indexes_buff);
	network.queue->enqueueWriteBuffer(*(network.cl_buff_training_set_id), CL_TRUE, 0, sizeof(cl_int), network.training_set_id_buff);
}

void Create_Kernels(kiv_ppr_network_gpu::TNetworkGPU& network, cl::Program program) {
	network.inc_train_set_id = new cl::Kernel(program, "inc_train_set_id");
	network.set_data_to_layers = new cl::Kernel(program, "set_data_to_layers");
	network.feed_forward_input_hidden1 = new cl::Kernel(program, "feed_forward_input_hidden1");
	network.feed_forward_hidden1_hidden2 = new cl::Kernel(program, "feed_forward_hidden1_hidden2");
	network.feed_forward_hidden2_output = new cl::Kernel(program, "feed_forward_hidden2_output");
	network.set_index_of_result = new cl::Kernel(program, "set_index_of_result");
	network.back_prop_output = new cl::Kernel(program, "back_prop_output");
	network.back_prop_hidden2 = new cl::Kernel(program, "back_prop_hidden2");
	network.back_prop_hidden1 = new cl::Kernel(program, "back_prop_hidden1");
	network.back_prop_input = new cl::Kernel(program, "back_prop_input");
	network.update_weights = new cl::Kernel(program, "update_weights");
}

void Set_Args_To_Kernels(kiv_ppr_network_gpu::TNetworkGPU& network) {

	// Kernel: inc_train_set_id
	network.inc_train_set_id->setArg(0, *(network.cl_buff_training_set_id));

	// Kernel: set_data_to_layers
	network.set_data_to_layers->setArg(0, *(network.cl_buff_training_set_id));
	network.set_data_to_layers->setArg(1, *(network.cl_buff_neural_net_data));
	network.set_data_to_layers->setArg(2, *(network.cl_buff_input_data));
	network.set_data_to_layers->setArg(3, *(network.cl_buff_helper_data));

	// Kernel: feed_forward_input_hidden1
	network.feed_forward_input_hidden1->setArg(0, *(network.cl_buff_neural_net_data));

	// Kernel: feed_forward_hidden1_hidden2
	network.feed_forward_hidden1_hidden2->setArg(0, *(network.cl_buff_neural_net_data));

	// Kernel: feed_forward_hidden2_output
	network.feed_forward_hidden2_output->setArg(0, *(network.cl_buff_neural_net_data));
	network.feed_forward_hidden2_output->setArg(1, *(network.cl_buff_helper_data));

	// Kernel: set_index_of_result
	network.set_index_of_result->setArg(0, *(network.cl_buff_training_set_id));
	network.set_index_of_result->setArg(1, *(network.cl_buff_neural_net_data));
	network.set_index_of_result->setArg(2, *(network.cl_buff_result_indexes));
	network.set_index_of_result->setArg(3, *(network.cl_buff_helper_data));

	// Kernel: back_prop_output
	network.back_prop_output->setArg(0, *(network.cl_buff_training_set_id));
	network.back_prop_output->setArg(1, *(network.cl_buff_neural_net_data));
	network.back_prop_output->setArg(2, *(network.cl_buff_target_data));
	network.back_prop_output->setArg(3, *(network.cl_buff_delta_gradient_data));

	// Kernel: back_prop_hidden2
	network.back_prop_hidden2->setArg(0, *(network.cl_buff_neural_net_data));
	network.back_prop_hidden2->setArg(1, *(network.cl_buff_delta_gradient_data));

	// Kernel: back_prop_hidden1
	network.back_prop_hidden1->setArg(0, *(network.cl_buff_neural_net_data));
	network.back_prop_hidden1->setArg(1, *(network.cl_buff_delta_gradient_data));

	// Kernel: back_prop_input
	network.back_prop_input->setArg(0, *(network.cl_buff_neural_net_data));
	network.back_prop_input->setArg(1, *(network.cl_buff_delta_gradient_data));

	// Kernel: update_weights
	network.update_weights->setArg(0, *(network.cl_buff_neural_net_data));
	network.update_weights->setArg(1, *(network.cl_buff_delta_gradient_data));
}

void kiv_ppr_network_gpu::Get_Relative_Errors_Vector(kiv_ppr_network_gpu::TNetworkGPU& network, std::vector<double>& expected_values, std::vector<double>& relative_errors_vector) {
	double result_value = 0.0;

	for (unsigned i = 0; i < network.num_of_training_sets; i++) {
		result_value = kiv_ppr_utils::Band_Index_To_Level(network.result_indexes_buff[i]);

		relative_errors_vector.push_back((std::abs(result_value - expected_values[i]) / expected_values[i]));
	}

}

void kiv_ppr_network_gpu::Train(kiv_ppr_network_gpu::TNetworkGPU& network) {

	for (unsigned i = 0; i < network.num_of_training_sets; i++) {
		//network.training_set_id_buff[0] = i;

		//network.queue->enqueueWriteBuffer(*(network.cl_buff_training_set_id), CL_TRUE, 0, sizeof(cl_int), network.training_set_id_buff);

		// Kernel: inc_train_set_id
		network.queue->enqueueNDRangeKernel(*(network.inc_train_set_id), cl::NullRange, cl::NDRange(1), cl::NullRange);
		//network.queue->finish();

		// Kernel: set_data_to_layers
		network.queue->enqueueNDRangeKernel(*(network.set_data_to_layers), cl::NullRange, cl::NDRange(OUTPUT_LAYER_NEURONS_COUNT), cl::NullRange);
		//network.queue->finish();

		// Kernel: feed_forward_input_hidden1
		network.queue->enqueueNDRangeKernel(*(network.feed_forward_input_hidden1), cl::NullRange, cl::NDRange(HIDDEN1_LAYER_NEURONS_COUNT), cl::NullRange);
		//network.queue->finish();

		// Kernel: feed_forward_hidden1_hidden2
		network.queue->enqueueNDRangeKernel(*(network.feed_forward_hidden1_hidden2), cl::NullRange, cl::NDRange(HIDDEN2_LAYER_NEURONS_COUNT), cl::NullRange);
		//network.queue->finish();

		// Kernel: feed_forward_hidden2_output
		network.queue->enqueueNDRangeKernel(*(network.feed_forward_hidden2_output), cl::NullRange, cl::NDRange(OUTPUT_LAYER_NEURONS_COUNT), cl::NullRange);
		//network.queue->finish();

		// Kernel: set_index_of_result
		network.queue->enqueueNDRangeKernel(*(network.set_index_of_result), cl::NullRange, cl::NDRange(OUTPUT_LAYER_NEURONS_COUNT), cl::NullRange);
		//network.queue->finish();

		// Kernel: back_prop_output
		network.queue->enqueueNDRangeKernel(*(network.back_prop_output), cl::NullRange, cl::NDRange(OUTPUT_LAYER_NEURONS_COUNT), cl::NullRange);
		//network.queue->finish();

		// Kernel: back_prop_hidden2
		network.queue->enqueueNDRangeKernel(*(network.back_prop_hidden2), cl::NullRange, cl::NDRange(HIDDEN2_LAYER_NEURONS_COUNT + 1), cl::NullRange);
		//network.queue->finish();

		// Kernel: back_prop_hidden1
		network.queue->enqueueNDRangeKernel(*(network.back_prop_hidden1), cl::NullRange, cl::NDRange(HIDDEN1_LAYER_NEURONS_COUNT + 1), cl::NullRange);
		//network.queue->finish();

		// Kernel: back_prop_input
		network.queue->enqueueNDRangeKernel(*(network.back_prop_input), cl::NullRange, cl::NDRange(INPUT_LAYER_NEURONS_COUNT + 1), cl::NullRange);
		//network.queue->finish();

		// Kernel: update_weights
		network.queue->enqueueNDRangeKernel(*(network.update_weights), cl::NullRange, cl::NDRange(OUTPUT_LAYER_NEURONS_COUNT), cl::NullRange);
		network.queue->finish();

		//network.queue->enqueueReadBuffer(*(network.cl_buff_neural_net_data), CL_TRUE, 0, sizeof(cl_float) * CL_BUFF_NEURAL_NET_DATA_SIZE, network.neural_net_buff);
		//network.queue->enqueueReadBuffer(*(network.cl_buff_result_indexes), CL_TRUE, 0, sizeof(cl_int) * (network.num_of_training_sets + 1), network.result_indexes_buff);

		/*
		for (unsigned i = 0; i < CL_BUFF_NEURAL_NET_DATA_SIZE; i++)
			std::cout << network.neural_net_buff[i] << std::endl;
		
		std::cout << "\n\nVysledek: " << network.result_indexes_buff[0] << std::endl;
		*/
		
		/*
		for (unsigned i = 0; i < (network.num_of_training_sets + 1); i++)
			std::cout << network.result_indexes_buff[i] << std::endl;
			*/


		//network.queue->enqueueReadBuffer(*(network.cl_buff_neural_net_data), CL_TRUE, 0, sizeof(cl_float) * CL_BUFF_NEURAL_NET_DATA_SIZE, network.neural_net_buff);

		/*
		for (unsigned i = 0; i < CL_BUFF_NEURAL_NET_DATA_SIZE; i++)
			std::cout << network.neural_net_buff[i] << std::endl;
		*/
	}

	network.queue->enqueueReadBuffer(*(network.cl_buff_result_indexes), CL_TRUE, 0, sizeof(cl_int) * (network.num_of_training_sets + 1), network.result_indexes_buff);

}

void kiv_ppr_network_gpu::Init_Data(kiv_ppr_network_gpu::TNetworkGPU& network, unsigned input_values_size, unsigned target_values_size) {
	Init_Network_Data(network);
	Init_Training_Set_Id(network);
	Init_Weights(network);

	Write_Init_Data_To_Buffers(network, input_values_size, target_values_size);
}

kiv_ppr_network_gpu::TNetworkGPU kiv_ppr_network_gpu::New_Network(std::vector<double>& input_values, std::vector<double>& target_values, unsigned num_of_training_sets) {
    kiv_ppr_network_gpu::TNetworkGPU new_network;
	std::vector<cl::Device> devices_gpu;
	std::vector<cl::Device> devices_cpu;
	bool prefer_gpu = true;
	std::string cl_code;

	cl::Program program;
	cl::Context context;

	unsigned input_values_size = input_values.size();
	unsigned target_values_size = target_values.size();

	new_network.num_of_training_sets = num_of_training_sets;

	if (!Find_Devices(devices_gpu, devices_cpu)) {
		new_network.is_valid = false;
		return new_network;
	}

	Choose_Device(new_network.default_device, devices_gpu, devices_cpu, prefer_gpu);

	std::cout << "\nChosen device: " << new_network.default_device->getInfo<CL_DEVICE_NAME>() << std::endl;
	std::cout << "\n----------------------------------------------\n\n";

	Load_Code(CL_FILE_DEST, cl_code);

	if (!Build_Cl_Program(new_network.default_device, context, program, cl_code)) {
		new_network.is_valid = false;
		return new_network;
	}

	Allocate_Buffers(new_network, context, input_values_size, target_values_size);

	Convert_Vectors_To_Buff(input_values, target_values, new_network.input_values_buff, new_network.target_values_buff);

	Create_Kernels(new_network, program);
	Set_Args_To_Kernels(new_network);

	new_network.queue = new cl::CommandQueue(context, *(new_network.default_device));

	new_network.is_valid = true;
		
	return new_network;
}

//TODO: uvolnit pamet po bufferech