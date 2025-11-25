import math
import torch
import numpy as np
import random
from scipy.ndimage import binary_erosion, binary_dilation

class EstimatorHighlightGenerator:
    def __init__(self, max_gain=10.0):
        self.max_gain = max_gain
    
    def gaussian_kernel(self, distance, radius, max_gain, boundary_gain):
        return max_gain * torch.exp(-(distance ** 2) / (2 * (radius ** 2)))
    
    def inverse_square_exponential_kernel(self, distance, radius, max_gain, boundary_gain):
        epsilon = 1.0
        beta = 1.0
        return (max_gain / ((distance / radius) ** 2 + epsilon)) * torch.exp(-beta * (distance / radius))
    
    def random_kernel(self, distance, radius, max_gain, boundary_gain):
        kernels = [self.gaussian_kernel, self.inverse_square_exponential_kernel]
        kernel = random.choice(kernels)
        kernel_mask = kernel(distance, radius, max_gain, boundary_gain)
        kernel_mask[kernel_mask < boundary_gain] = boundary_gain
        return kernel_mask
    
    def torch_line(self, mask, x1, y1, x2, y2):
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        while True:
            if 0 <= x1 < mask.shape[1] and 0 <= y1 < mask.shape[0]:
                mask[y1, x1] = 1
            if x1 == x2 and y1 == y2:
                break
            e2 = err * 2
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
    
    def generate_highlight(self, tensor):
        tensor = tensor.unsqueeze(0)
        H, W = tensor.shape[2], tensor.shape[3]
        mask = torch.ones((H, W), dtype=torch.uint8)
        addmap = torch.zeros((H, W))
        final_image = torch.clone(tensor)

        total_area = 0.0
        max_area = (H * W) / 5.0
        
        region_type = torch.randint(0, 2, (1,)).item()
        center_x = torch.randint(0, W, (1,)).item()
        center_y = torch.randint(0, H, (1,)).item()

        if region_type == 0:
            radius = torch.randint(10, 50, (1,)).item()
            area = math.pi * (radius ** 2)
        else:
            num_sides = torch.randint(5, 9, (1,)).item()
            radius = torch.randint(10, 50, (1,)).item()
            area = num_sides * (radius ** 2) * 0.5

        region_gain = torch.rand(1).item() * self.max_gain
        region_gain_matrix = torch.ones((H, W))

        if region_type == 0:
            y_grid, x_grid = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
            distance = ((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2).sqrt()
            region_mask = distance <= radius
            boundary_gain = min(random.randint(1, 1), self.max_gain)
            region_gain_matrix[region_mask] = self.random_kernel(
                distance[region_mask], radius, region_gain, boundary_gain
            )
        else:
            num_sides = max(3, num_sides)
            angles = torch.linspace(0, 2 * math.pi, num_sides + 1)
            x_offsets = (torch.cos(angles) * radius).int()
            y_offsets = (torch.sin(angles) * radius).int()
            polygon_points = [(center_x + x_offsets[i], center_y + y_offsets[i]) for i in range(num_sides)]
            region_mask = torch.zeros((H, W), dtype=torch.bool)

            for i in range(num_sides):
                x1, y1 = polygon_points[i]
                x2, y2 = polygon_points[(i + 1) % num_sides]
                self.torch_line(region_mask, x1, y1, x2, y2)

            y_grid, x_grid = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
            distance_from_center = ((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2).sqrt()
            boundary_gain = min(random.randint(1, 1), self.max_gain)
            region_gain_matrix[region_mask] = self.random_kernel(
                distance_from_center[region_mask], radius, region_gain, boundary_gain
            )

        mask[region_mask] = 0
        final_image += region_gain_matrix.unsqueeze(0).unsqueeze(0) * region_mask.float()
        region_gain_matrix1 = torch.ones_like(region_gain_matrix)
        region_gain_matrix1[region_mask] = region_gain_matrix[region_mask]
        addmap = region_gain_matrix1.float()
        total_area += area

        final_image = torch.clamp(final_image, 0, 1)
        return mask, addmap, final_image
    

class DenoiserHighlightGenerator:
    def __init__(self, max_gain=10.0):
        self.max_gain = max_gain
    
    def gaussian_kernel(self, distance, radius, max_gain, boundary_gain):
        alpha = random.uniform(0.1, 1.0)
        return max_gain * torch.exp(-(distance ** 2) / (alpha * (radius ** 2)))
    
    def inverse_square_exponential_kernel(self, distance, radius, max_gain, boundary_gain):
        epsilon = 1.0
        beta = random.uniform(1.0, math.sqrt(max_gain))
        return (max_gain / ((((distance / radius) ** 2) * beta) + epsilon))
    
    def random_kernel(self, distance, radius, max_gain, boundary_gain):
        kernels = [self.gaussian_kernel, self.inverse_square_exponential_kernel]
        kernel = random.choice(kernels)
        kernel_mask = kernel(distance, radius, max_gain, boundary_gain)
        kernel_mask[kernel_mask < boundary_gain] = boundary_gain
        return kernel_mask
    
    def torch_line_fill(self, mask, x1, y1, x2, y2):
        H, W = mask.shape
        x1, x2, y1, y2 = min(x1, x2), min(max(x1, x2), W-1), min(y1, y2), min(max(y1, y2), H-1)
        a = random.randint(1, x2-x1+2)
        b, c = random.uniform(-1, 1), random.uniform(-1, 1)
        
        for i in range(x1, x2):
            ymin = torch.tensor(min(np.floor(y1 + b * y1 * np.sin(i/a*np.pi)), H-2)).int()
            ymax = torch.tensor(min(np.floor(y2 + c * y2 * np.cos(i/a*np.pi)), H-2)).int()
            mask[ymin:ymax, i] = 1
    
    def generate_highlight(self, tensor):
        tensor = tensor.unsqueeze(0)
        H, W = tensor.shape[2], tensor.shape[3]
        mask = torch.ones((H, W), dtype=torch.uint8)
        addmap = torch.zeros((H, W))
        final_image = torch.clone(tensor)

        total_area = 0.0
        max_area = (H * W) / 2.0
        num_regions = torch.randint(1, 11, (1,)).item()

        region_gain_matrix = torch.ones((H, W))
        region_gain_matrix1 = torch.ones_like(region_gain_matrix)
        
        for _ in range(num_regions):
            center_x = torch.randint(0, W, (1,)).item()
            center_y = torch.randint(0, H, (1,)).item()
            radius = torch.randint(10, 300, (1,)).item()
            area = math.pi * (radius ** 2)

            if total_area + area > max_area:
                continue

            region_gain = torch.rand(1).item() * self.max_gain
            
            num_sides = torch.randint(5, 9, (1,)).item()
            num_sides = max(3, num_sides)
            
            x1, x2 = random.randint(0, W), random.randint(0, W)
            y1, y2 = random.randint(0, H), random.randint(0, H)
            center_x, center_y = random.randint(min(x1, x2), max(x1, x2)), random.randint(min(y1, y2), max(y1, y2))

            region_mask = torch.zeros((H, W), dtype=torch.bool)
            self.torch_line_fill(region_mask, x1, y1, x2, y2)

            region_mask_np = region_mask.detach().cpu().numpy()
            for _ in range(random.randint(0, 5)):
                if random.choice([True, False]):
                    region_mask_np = binary_erosion(region_mask_np)
                else:
                    region_mask_np = binary_dilation(region_mask_np)
            region_mask = torch.from_numpy(region_mask_np).to(region_mask.device)

            y_grid, x_grid = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
            distance_from_center = ((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2).sqrt()

            boundary_gain = min(random.randint(1, 1), self.max_gain)
            region_gain_matrix[region_mask] = self.random_kernel(
                distance_from_center[region_mask], radius, region_gain, boundary_gain
            )

            mask[region_mask] = 0
            final_image += region_gain_matrix.unsqueeze(0).unsqueeze(0) * region_mask.float()
            
            region_gain_matrix1[region_mask] = region_gain_matrix[region_mask]
            addmap = region_gain_matrix1.float()
            total_area += area

        final_image = torch.clamp(final_image, 0, 1)
        return mask, addmap, final_image
    
