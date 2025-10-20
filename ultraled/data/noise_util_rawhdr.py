import torch
from torch import nn
import random
from ultraled.data.commom_noise_util import *



class NoiseGenerator(nn.Module):
    def __init__(self, camera_params, noise_type, *, engine='torch') -> None:
        super().__init__()
        self.camera_params = camera_params
        self.cameras = list(camera_params.keys())
        print('Current Using Cameras: ', self.cameras)

        if engine == 'numpy':
            self.engine = NumpyEngine()
        else:
            self.engine = TorchEngine()

        self.noise_type = noise_type.lower()
        self.read_type = 'TurkeyLambda' if 't' in self.noise_type else \
            ('Gaussian' if 'g' in self.noise_type else None)

    @property
    def sample_K(self):
        index = self.engine.randint(0, len(self.camera_params))
        self.current_camera = self.cameras[index]
        self.current_camera_params = self.camera_params[self.current_camera]
        self.current_k_range = [
            self.camera_params[self.current_camera]['Kmin'],
            self.camera_params[self.current_camera]['Kmax']
        ]
        log_K_max = self.engine.log(self.current_camera_params['Kmax'])
        log_K_min = self.engine.log(self.current_camera_params['Kmin'])
        log_K = self.engine.uniform(log_K_min, log_K_max)
        self.log_K = log_K
        return self.engine.exp(log_K)


    @property
    def sample_read_param(self):
        slope = self.current_camera_params[self.read_type]['slope']
        bias = self.current_camera_params[self.read_type]['bias']
        sigma = self.current_camera_params[self.read_type]['sigma']
        mu = self.log_K * slope + bias
        sample = self.engine.randn() * sigma + mu
        return self.engine.exp(sample)

    @property
    def sample_turkey_lambda(self):
        if self.read_type != 'TurkeyLambda':
            return None
        index = self.engine.randint(0, len(self.current_camera_params[self.read_type]['lambda']))
        return self.current_camera_params[self.read_type]['lambda'][index]

    @property
    def sample_row_param(self):
        slope = self.current_camera_params['Row']['slope']
        bias = self.current_camera_params['Row']['bias']
        sigma = self.current_camera_params['Row']['sigma']
        mu = self.log_K * slope + bias
        sample = self.engine.randn() * sigma + mu
        return self.engine.exp(sample)



    @property
    def sample_color_bias(self):
        count = len(self.current_camera_params['ColorBias'])
        i_range = (self.current_k_range[1] - self.current_k_range[0]) / count
        index = int((self.engine.exp(self.log_K) - self.current_k_range[0]) // i_range)
        index = max(min(index, len(self.current_camera_params['ColorBias']) - 1), 0)
        color_bias = self.current_camera_params['ColorBias'][index]
        return self.engine.to_engine_type(color_bias).reshape(4, 1, 1)
    
    

    @torch.no_grad()
    # def forward(self, img):
    def forward(self, img, *, K=None):


        if K is not None:
            self.sample_K = K
        else:
            K = self.sample_K
        # K = self.sample_K

        noise1 = []
        # possion noise
        if 'p' in self.noise_type:
            shot_noise = self.engine.shot_noise(img, K)
            noise1.append(shot_noise)
        # read noise
        if 'g' in self.noise_type:
            read_noise = self.engine.gaussian_noise(img, self.sample_read_param)
            noise1.append(read_noise)
        elif 't' in self.noise_type:
            read_noise = self.engine.turkey_lambda_noise(img, self.sample_read_param, self.sample_turkey_lambda)
            noise1.append(read_noise)
        # row noise
        if 'r' in self.noise_type:
            row_noise = self.engine.row_noise(img, self.sample_row_param)
            noise1.append(row_noise)
        # quant noise
        if 'q' in self.noise_type:
            quant_noise = self.engine.quant_noise(img, 1)
            noise1.append(quant_noise)
        if 'c' in self.noise_type:
            noise1.append(self.sample_color_bias.to(img.device))


        return img, noise1




























### Support multiple cameras

# class NoiseGenerator(nn.Module):
#     def __init__(self, camera_params, noise_type, *, engine='torch') -> None:
#         super().__init__()
#         self.camera_params = camera_params
#         self.cameras = list(camera_params.keys())
#         print('Current Using Cameras: ', self.cameras)

#         if engine == 'numpy':
#             self.engine = NumpyEngine()
#         else:
#             self.engine = TorchEngine()

#         self.noise_type = noise_type.lower()
#         self.read_type = 'TurkeyLambda' if 't' in self.noise_type else \
#             ('Gaussian' if 'g' in self.noise_type else None)

#     @property
#     def sample_K(self):
#         index = self.engine.randint(0, len(self.camera_params))
#         self.current_camera = self.cameras[index]
#         self.current_camera_params = self.camera_params[self.current_camera]
#         self.current_k_range = [
#             self.camera_params[self.current_camera]['Kmin'],
#             self.camera_params[self.current_camera]['Kmax']
#         ]
#         log_K_max = self.engine.log(self.current_camera_params['Kmax'])
#         log_K_min = self.engine.log(self.current_camera_params['Kmin'])
#         log_K = self.engine.uniform(log_K_min, log_K_max)
#         self.log_K = log_K
#         return self.engine.exp(log_K)

#     @sample_K.setter
#     def sample_K(self, K):
#         assert len(self.camera_params) == 1
#         index = 0
#         self.current_camera = self.cameras[index]
#         self.current_camera_params = self.camera_params[self.current_camera]
#         self.current_k_range = [
#             self.camera_params[self.current_camera]['Kmin'],
#             self.camera_params[self.current_camera]['Kmax']
#         ]
#         self.log_K = self.engine.log(K)

#     @property
#     def sample_read_param(self):
#         slope = self.current_camera_params[self.read_type]['slope']
#         bias = self.current_camera_params[self.read_type]['bias']
#         sigma = self.current_camera_params[self.read_type]['sigma']
#         mu = self.log_K * slope + bias
#         sample = self.engine.randn() * sigma + mu
#         return self.engine.exp(sample)

#     @property
#     def sample_turkey_lambda(self):
#         if self.read_type != 'TurkeyLambda':
#             return None
#         index = self.engine.randint(0, len(self.current_camera_params[self.read_type]['lambda']))
#         return self.current_camera_params[self.read_type]['lambda'][index]

#     @property
#     def sample_row_param(self):
#         slope = self.current_camera_params['Row']['slope']
#         bias = self.current_camera_params['Row']['bias']
#         sigma = self.current_camera_params['Row']['sigma']
#         mu = self.log_K * slope + bias
#         sample = self.engine.randn() * sigma + mu
#         return self.engine.exp(sample)

#     @property
#     def sample_color_bias(self):
#         count = len(self.current_camera_params['ColorBias'])
#         i_range = (self.current_k_range[1] - self.current_k_range[0]) / count
#         index = int((self.engine.exp(self.log_K) - self.current_k_range[0]) // i_range)
#         index = max(min(index, len(self.current_camera_params['ColorBias']) - 1), 0)
#         color_bias = self.current_camera_params['ColorBias'][index]
#         return self.engine.to_engine_type(color_bias).reshape(4, 1, 1)

#     @torch.no_grad()
#     # def forward(self, img):
#     def forward(self, img, *, K=None):
#         if K is not None:
#             self.sample_K = K
#         else:
#             K = self.sample_K
#         # K = self.sample_K

#         noise1 = []
#         # possion noise
#         if 'p' in self.noise_type:
#             shot_noise = self.engine.shot_noise(img, K)
#             noise1.append(shot_noise)
#         # read noise
#         if 'g' in self.noise_type:
#             read_noise = self.engine.gaussian_noise(img, self.sample_read_param)
#             noise1.append(read_noise)
#         elif 't' in self.noise_type:
#             read_noise = self.engine.turkey_lambda_noise(img, self.sample_read_param, self.sample_turkey_lambda)
#             noise1.append(read_noise)
#         # row noise
#         if 'r' in self.noise_type:
#             row_noise = self.engine.row_noise(img, self.sample_row_param)
#             noise1.append(row_noise)
#         # quant noise
#         if 'q' in self.noise_type:
#             quant_noise = self.engine.quant_noise(img, 1)
#             noise1.append(quant_noise)
#         if 'c' in self.noise_type:
#             noise1.append(self.sample_color_bias)

#         return img, noise1

