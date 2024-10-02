# 100_days_of_computer_vision

This repository contains the codes for **100 days of computer vision** challenge. I pursue this challenge with a focus on concepts of 3D & 2D Computer vision (and thus, some concepts in computer graphics). The challenge is inspired by the [100 Days of Code](https://www.100daysofcode.com/) challenge. 

## Covered Topics
1. __PyTorch3D__:
   
   Please refer to [this readme file](Pytorch3D/README.md) for in-depth information on the implemented projects with PyTorch3D. The covered projects are:

      1. [Rendering a textured mesh](Pytorch3D/notebooks/render_textured.ipynb) 

         ![Texture](Images/texture_1.png)

         ![Texture UV map](Images/texture_2.png)

         ![Textured 3D Cow](Images/cow_3d.png)

      3. [Rendering a dense pose](Pytorch3D/notebooks/render_dense_pose.ipynb)

         ![Dense Pose Texture](Images/dense_tex.png)

         ![Dense Pose](Images/dense_pose.png)

      5. [Deforming a base mesh and optimizing to get to the target mesh](Pytorch3D/notebooks/deform_to_fit.ipynb)

         ![Source Point cloud](Images/source_pc.png)

         ![Target Point Cloud](Images/target_pc.png)

         ![Losses and metrics](Images/losses_deformed.png)
         
         ![Deformed source point cloud](Images/deformed_source_pc.png)
      

2. __From Scratch Projects__:
  
    Please refer to [this readme file](FromScratch/README.md) for in-depth information on the implemented projects from scratch. The covered projects are:

   1. [Image Formation](FromScratch/Image_Formation/main.py)
      ![Image Formation 3D Plot](Images/formation_3d.png)
      ![Image Formation 2D Plot](Images/formation_2d.png)
   
   2. [ViT from scratch](FromScratch/ViT/main.ipynb)
   
      Making patches as tokens from an input image:

      ![Patches of an image](Images/VIT_Patches.png)

      Visualizing Positional Embedding:

      ![Positional Embedding](Images/VIT_Posenc.png)

      Testing the predictions of classifier:

      ![Testing Classifier](Images/VIT_test.png)

   3. [VAE From Scratch](FromScratch/VAE/main.ipynb)

      A sample from the final trained VAE (on Mnist):

      ![Number 6 Generated](Images/generated_6_VAE.png)

      Generating 25 samples:

      ![25 Samples](Images/generated_25_samples_vae.png)

      Visualizing the distribution of generation in the trained VAE:

      ![Distribution](Images/distribution_of_generation_vae.png)

   4. [Spherical Harmonics From Scratch](FromScratch/Spherical_Harmonics/main.ipynb)

      Vectors on the surface of a sphere:

      ![Vectors SH](Images/sphere_vectors_SH.png)

      Legendre Polynomials:

      ![Legendre Polynomials](Images/Legendre_Polynomials_SH.png)

      Spherical Harmonics with $l=1$ and $m=-1$:
      
      https://github.com/user-attachments/assets/251683a4-0f1a-4ac0-9b97-92cde95ea6fd
      
      Spherical Harmonics with $l=1$ and $m=0$:
      
      https://github.com/user-attachments/assets/6d7f1fa8-091c-4103-bf85-f82dc0dc3bce
      
      Spherical Harmonics with $l=1$ and $m=1$:

      https://github.com/user-attachments/assets/191b21d4-434a-4632-9cc4-f08b64f41842

      
   5. [Grid Encoding (like InstantNGP) From Scratch](FromScratch/Grid_Encoding/3d_grid.ipynb)
   
      A sample 3D grid (sampled points):

      ![3D Grid](Images/3d_grid_base.png)

      A sample point $\bar{x}=(x, y, z)$ and it's nearest voxel:

      ![Nearest Voxel](Images/random_point_in_3d_grid.png)

      Then the point's encoding is calculated as the __Linear interpolation__ of the 8 nearest voxels. For the training process, please refer to the [notebook](FromScratch/Grid_Encoding/3d_grid.ipynb).

   6. [RGB Prediction Using Spherical Harmonics](FromScratch/Spherical_Harmonics/main.ipynb)

      The RGB prediction using Spherical Harmonics:
      
   7. [Tiny Renderer from Scratch](FromScratch/Tiny%20Renderer/main.cpp)
   
      This is a C++ implementation of a tiny renderer. I made it to play around with the concepts of comuter graphics and the most important focus here is __rasterization__. I took the guidances of the [tinyrenderer](github.com/ssloy/tinyrenderer) repository.

      Rendering simple line segments (without anti-aliasing):
      ```Cpp
      void line(int x0, int y0, int x1, int y1, TGAImage &image, TGAColor color);
      ```

      ![Line Segment](Images/line_output.jpeg)

      Rendering an `obj` model:
      ```Cpp
      void wireframe_obj(Model &model, TGAImage &image, TGAColor color);
      ```

      ![Obj wireframe](Images/obj_output.jpeg)

      Rasterizing a triangle (only contours):
      ```Cpp
      void triangle_contour(Vec2f t0, Vec2f t1, Vec2f t2, TGAImage &image,
                      TGAColor color);
      ```

      ![Triangle Contour](Images/contour_output.jpeg)

      Rasterizing a triangle (filled):
      ```Cpp
      void triangle(Vec2i t0, Vec2i t1, Vec2i t2, TGAImage &image, TGAColor color);
      ```

      ![Triangle Filled](Images/filled_output.jpeg)

   8. [Diffusion model from scratch](FromScratch/Diffusion/main.py)
   ...
