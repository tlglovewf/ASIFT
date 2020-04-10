Erie
====

Affine matching C++ library for OpenCV (based on ASIFT)

#### Algorithm

Algorithm is based on AffineSIFT [1]. Given a pair of images, a number of 'views' for each of them are generated. A view is the original image distorted by an affine transformation, it shows how the image would look like from some preset viewpoint. Usually it is enough to have 10 views (9 distortions + 1 original) or less for both images. All 10 views from the 1st image are pairwise matched to all 10 views of the second image. `feature2d` module of OpenCV is used for matching. If those pairs that have enough keypoint matches, they contribute to the result. This is roughly the idea.


#### Differences from ASIFT

1. Features other that SIFT can be used (ORB, SURF, BRISK, ...)
2. Incremental matching is implemented. If images can not be matched without distortions, tilt = 1 is used to create distorted 'views'. If that is not enough, tilt = 2 is used, and so on.

### Compile

make sure you have `OpenCV` and `Boost` libraries

```
cd Eerie
mkdir build
cd build
cmake ..
make
cd ..
```

### Usage

```
# a minimal working example
build/aff_demo

# matching two images (--help for options)
build/aff_match_images --help
```



#### References

[1] J.M. Morel and G.Yu, ASIFT: A New Framework for Fully Affine Invariant Image Comparison, SIAM Journal on Imaging Sciences, vol. 2, issue 2, 2009.
