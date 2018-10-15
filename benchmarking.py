import torch, time
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_duration_mean_ms(arr):
    arr = np.array(arr)
    return arr.mean() * 1000

def run_benchmarks(arr, batch_size, n_iters):

    for m in arr:
        print("*"*10, "Model {}".format(m), "*"*10)
        forward_pass_time = []
        backward_pass_time = []
        # total_time = []
        inference_time = []
        model = m().to(device)
        input_tensor = torch.randn(batch_size, 3, 224, 224).to(device)
        grad_tensor = torch.ones(batch_size, 1000).to(device)

        for i in range(n_iters):

            start_time_forward = time.time()
            result = model(input_tensor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time_forward = time.time()

            start_time_backward = time.time()
            result.backward(grad_tensor)        
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time_backward = time.time()

            if i is 1: # Ignore first iteration
                continue
            forward_pass_time.append(end_time_forward - start_time_forward)
            backward_pass_time.append(end_time_backward - start_time_backward)
            # total_time.append(forward_pass_time[-1] + backward_pass_time[-1])

        forward_pass_time_mean = get_duration_mean_ms(forward_pass_time)
        backward_pass_time_mean = get_duration_mean_ms(backward_pass_time)
        total_time = forward_pass_time_mean + backward_pass_time_mean
        print("Forward Pass Time : {}ms".format(round(forward_pass_time_mean, 1)))
        print("Backward Pass Time : {}ms".format(round(backward_pass_time_mean, 1)))
        print("Total Time : {}ms".format(round(total_time, 1)))

        model.eval()
        for i in range(n_iters):
            start_time_inference = time.time()
            result = model(input_tensor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time_inference = time.time()

            if i is 1: # Ignore first iteration
                continue
            inference_time.append(end_time_forward - start_time_forward)

        inference_time_mean = get_duration_mean_ms(inference_time)
        print("Inference Time : {}ms".format(round(inference_time_mean,1)))
