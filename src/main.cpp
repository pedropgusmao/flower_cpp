#include <iostream>
#include <torch/torch.h>
#include <torchvision/vision.h>
#include <torchvision/models/resnet.h>
#include "cifar10.h"

template <typename DataLoader>
void test(vision::models::ResNet18 &net,
          DataLoader &test_loader,
          torch::Device device)
{
  net->to(device);
  net->eval();
  size_t num_samples = 0;
  size_t num_correct = 0;
  float running_loss = 0.0;

  // Iterate the data loader to yield batches from the dataset.
  for (auto &batch : *test_loader)
  {
    auto data = batch.data.to(device), target = batch.target.to(device);
    num_samples += data.size(0);
    // Execute the model on the input data.
    torch::Tensor output = net->forward(data);

    // Compute a loss value to judge the prediction of our model.
    torch::Tensor loss = torch::nn::functional::cross_entropy(output, target);

    AT_ASSERT(!std::isnan(loss.template item<float>()));
    running_loss += loss.item<float>() * data.size(0); // CHECK IF IT IS DOUBLE OR FLOAT!!!!
    auto prediction = output.argmax(1);
    num_correct += prediction.eq(target).sum().template item<int64_t>();
  }
  auto sample_mean_loss = running_loss / num_samples;
  auto accuracy = static_cast<double>(num_correct) / num_samples;

  std::cout << "Testset - Loss: " << sample_mean_loss << ", Accuracy: " << accuracy << std::endl;
}

template <typename DataLoader>
void train(vision::models::ResNet18 &net, int64_t num_epochs,
           DataLoader &train_loader,
           torch::optim::Optimizer &optimizer,
           torch::Device device)
{
  net->to(device);
  net->train();
  for (size_t epoch = 1; epoch <= num_epochs; ++epoch)
  {
    size_t num_samples = 0;
    size_t num_correct = 0;
    float running_loss = 0.0;

    // Iterate the data loader to yield batches from the dataset.
    for (auto &batch : *train_loader)
    {
      auto data = batch.data.to(device), target = batch.target.to(device);
      num_samples += data.size(0);
      // Reset gradients.
      optimizer.zero_grad();
      // Execute the model on the input data.
      torch::Tensor output = net->forward(data);

      // Compute a loss value to judge the prediction of our model.
      torch::Tensor loss = torch::nn::functional::cross_entropy(output, target);

      AT_ASSERT(!std::isnan(loss.template item<float>()));
      //std::cout << loss.item<float>() << std::endl;
      running_loss += loss.item<float>() * data.size(0); // CHECK IF IT IS DOUBLE OR FLOAT!!!!
      auto prediction = output.argmax(1);
      num_correct += prediction.eq(target).sum().template item<int64_t>();

      // Compute gradients of the loss w.r.t. the parameters of our model.
      loss.backward();
      // Update the parameters based on the calculated gradients.
      optimizer.step();
    }
    auto sample_mean_loss = running_loss / num_samples;
    auto accuracy = static_cast<double>(num_correct) / num_samples;

    std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
              << sample_mean_loss << ", Accuracy: " << accuracy << std::endl;
  }
}

int main()
{
  // Check if we can work with GPUs
  auto cuda_available = torch::cuda::is_available();
  torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
  std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

  int num_classes = 10;
  auto net = vision::models::ResNet18(num_classes);

  // Load CIFAR10 Dataset
  int64_t kTrainBatchSize = 64;
  int64_t kTestBatchSize(kTrainBatchSize);
  const std::string CIFAR10_DATASET_PATH = "/home/pedro/repos/flower_cpp/data/cifar-10-batches-bin/";
  std::vector<double> norm_mean = {0.4914, 0.4822, 0.4465};
  std::vector<double> norm_std = {0.247, 0.243, 0.261};
  auto train_dataset = CIFAR10(CIFAR10_DATASET_PATH, CIFAR10::Mode::kTrain)
                           .map(torch::data::transforms::Normalize<>(norm_mean, norm_std))
                           .map(torch::data::transforms::Stack<>());
  const size_t train_dataset_size = train_dataset.size().value();
  auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_dataset), kTrainBatchSize);

  auto test_dataset = CIFAR10(CIFAR10_DATASET_PATH, CIFAR10::Mode::kTest)
                          .map(torch::data::transforms::Normalize<>(norm_mean, norm_std))
                          .map(torch::data::transforms::Stack<>());
  const size_t test_dataset_size = test_dataset.size().value();
  auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(test_dataset), kTestBatchSize);

  // Instantiate an SGD optimization algorithm to update our Net's parameters.
  float lr = 0.1;
  torch::optim::SGD optimizer(net->parameters(), lr);

  // Train
  int64_t num_epochs = 1;
  train(net, num_epochs, train_loader, optimizer, device);
  test(net, test_loader, device);

  // Serialize
  std::ostringstream stream;
  torch::save(net, stream);
  std::string str = stream.str();
  const char *chr = str.c_str();
  // Probably send the contents of chr.
  // This is a ZIP file. We need to read it as such in Python.

  //
  torch::save(net, "mytensor.pt");
}
