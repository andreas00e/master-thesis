import torch 

def main():
    test_tensor = torch.rand((5, 5, 5, 5))
    print(test_tensor.shape[:-2] + (test_tensor.shape[-2] * test_tensor.shape[-1], ))
    # print(len((test_tensor.shape[-2] * test_tensor.shape[-1],)))


if __name__ == '__main__': 
    main()