# BADA 
Burnt Area Detection Algorithm v0.1 

## Notes

- Stick with the RTS formulation
- Remember that with the assumptions of band independence we can speed the thing up to be raster based! This relies on the ability to do inversion of the covariance matrix really quickly because they are all three by three so we can add just apply a numpy function to each pixel and band as a ufunc style thing. The algorithm would then be crazy quick!

