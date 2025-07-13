use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::tensor::TensorHandle;

// Implementation follows GPU_dbl_blocked_houseqr method implementation
// https://github.com/janverschelde/PHCpack/blob/master/src/GPU/Matrices/dbl_baqr_kernels.cu
/*
#[cube(launch_unchecked)]
fn dbl_small_house<F: Float>( double *x0, double *x1, int dim, int dimline_indexlog2, double *v, double *beta ) {
   let j = UNIT_POS_X;

   let mut shared_v = SharedMemory::<F>::new(32 * 32);
   let mut product = SharedMemory::<F>::new(32 * 32);

   let mut stopflag = false;

   shared_v[j] = x1[j];              // reading of vector into shared memory
   product[j] = shared_v[j]*shared_v[j];      // for the 2-norm computation

   v[j+1] = shared_v[j];             // copies x to v, in case beta is zero
   if j == 0 {
      v[0] = F::from_int(1);
   } 

   sync_units();
   let pow_of_two = 1;                          // sum reduction
   for k in 0..dimline_indexlog2
   {
      if (j%(pow_of_two*2)) == 0 && (j+pow_of_two < dim) {
        product[j] = product[j] + product[j+pow_of_two];
      }
      pow_of_two = pow_of_two*2;
      sync_units();
   }
   // thread 0 computes the sqrt of the inner product, others wait
   if j == 0 {
      if product[0] == 0.0 {          // product[0] is sigma of house
         *beta = F::from_int(0);
         stopflag = true;
      }
   }
   sync_units();
   if stopflag {
    terminate!();             // case when sigma is zero
   } 
   if(j == 0) {     // thread zero sets beta
      let mu = F::sqrt((*x0)*(*x0) + product[0]);
      let v0 = if *x0 <= 0.0 {
         *x0 - mu
      } else {
         -product[0]/(*x0 + mu)
      };

      let v0p2 = v0*v0;
      *beta = 2.0*v0p2/(product[0] + v0p2);
      product[0] = v0;                         // v0 needed for normalization
   }
   sync_units();
   if *beta != 0.0 {
     shared_v[j] = shared_v[j]/product[0];
   }
   sync_units();
   v[j+1] = shared_v[j];
   if j == 0 {
     v[0] = F::from_int(1);
   }
}

fn GPU_dbl_small_house<R: Runtime, F: Float>(rows: u32, cols: u32, size_tiles: u32, num_tiles: u32,
   col_index: u32, rows_1: u32, k: u32, line_index: u32,
   r: &TensorHandleRef<'_, R>,
   v: &TensorHandleRef<'_, R>,
   beta: &TensorHandleRef<'_, R>)
{
    let rows_log2 = (rows_1 as f32).log2().ceil();
    let row_index = col_index * (rows + 1); // start of number in A_h
    let v_rows = rows - k * size_tiles;  // dimension of V matrix

    println!("rows: {rows} v_rows: {v_rows} cols: {cols} size_tiles: {size_tiles} num_tiles{num_tiles}");
    println!("k: {k} line_index: {line_index} rows_1 {rows_1} col_index {col_index} row_index {row_index}");

   if line_index > 0 {
      for i in 0..line_index {
         v_h[i] = 0.0; // insert zeros
      }
      cudaMemcpy(&V_d[line_index*nVrows],v_h,line_index*sizeof(double),
                 cudaMemcpyHostToDevice);
   }
   if rows_1 == 0 {
      beta_h[line_index] = F::from_int(0);
      v_h[0] = F::from_int(1);
      cudaMemcpy(&beta_d[line_index],&beta_h[line_index],sizeof(double),cudaMemcpyHostToDevice);
      cudaMemcpy(&V_d[line_index*nVrows+line_index],v_h,sizeof(double),cudaMemcpyHostToDevice);
   } else {
      dbl_small_house<<<1,rows1>>>
         (&A_d[rowidx],&A_d[rowidx+1],rows1,nrline_indexog2,
          &V_d[line_index*nVrows+line_index],&beta_d[line_index]);
   }
   cudaMemcpy(&beta_h[line_index],&beta_d[line_index],sizeof(double),cudaMemcpyDeviceToHost);
   if(verbose) {
      const size_t szhouse = nVrows*sizeof(double);

      cudaMemcpy(v_h,&V_d[line_index*nVrows],szhouse,cudaMemcpyDeviceToHost);
      cout << scientific << setprecision(16)
           << "beta[" << colidx << "] : " << beta_h[line_index] << endl;
      for(int i=0; i<nVrows; i++)
         cout << "v[" << i << "] : " << v_h[i] << endl;
   }
}
   */

/// line_indexaunch
pub fn launch_ref<R: Runtime, E: Float + CubeElement>(
    client: &ComputeClient<R::Server, R::Channel>,
    q: &TensorHandleRef<'_, R>,
    r: &TensorHandleRef<'_, R>,
) {
    launch::<R, E>(client, q, r);
}

pub fn launch<R: Runtime, E: Float + CubeElement>(
    client: &ComputeClient<R::Server, R::Channel>,
    q: &TensorHandleRef<'_, R>,
    r: &TensorHandleRef<'_, R>,
) {
    let num_tiles = 1; 
    let size_tiles = 32;
    let rows = q.shape[0];
    let cols = q.shape[1];

    let beta = TensorHandle::<R, E>::zeros(client, [size_tiles].to_vec());
    let v = TensorHandle::<R, E>::zeros(client, [rows].to_vec());
    // k runs over the number of blocks
    for k in 0..num_tiles {

       println!("Tile k = {k} out of {num_tiles}");
       /*
       // line runs over the columns in one block
       for line_index in 0..size_tiles {
        let col_index = k * size_tiles + line_index; // index of the current column
        let rows_1 = rows - col_index - 1; // #rows in Householder vector - 1
        GPU_dbl_small_house (rows,cols,size_tiles,num_tiles,col_index,rows_1, k, line_index,
         &r.as_ref(), &v.as_ref(), beta.as_ref());

          if beta[col_index] == F::from_int(0.0) {
             println!("Zero beta detected.");
          } else {
             if rows - col_index <= size_tiles {
                GPU_dbl_small_leftRupdate
                   (rows,cols,size_tiles,col_index,k,line_index,A_h,A_d,V_d,beta_h,beta_d,
                    tileRlapms,addcnt,mulcnt,verbose);
             } else {
                GPU_dbl_medium_leftRupdate
                   (rows,cols,size_tiles,colidx,k,line_index,A_h,A_d,V_d,beta_h,beta_d,
                    RTdotv_h,RTdotv_d,bRTv_h,bRTv_d,
                    RTvlapms,tileRlapms,addcnt,mulcnt,verbose);
             }
          }
       }
       GPU_dbl_medium_VB_to_W
          (rows,size_tiles,size_tiles,k,V_h,V_d,W_h,W_d,WYT_h,WYT_d,beta_h,beta_d);
       // update Q, WYT matrix has rows - k*size_tiles instead of rows
       GPU_dbl_small_QWYT
          (rows,size_tiles,k,Q_d,WYT_d,QWYT_d,QWYT_h,Q_h );
       GPU_dbl_small_Qupdate
          (rows,size_tiles,k,Q_d,QWYT_d,Q_h,Qaddlapms);
       if k < num_tiles-1 { // update R
          GPU_dbl_small_YWT
             (rows,size_tiles,k,V_d,W_d,YWT_d,YWT_h);
          GPU_dbl_small_YWTC
             (rows,cols,size_tiles,k,YWT_d,A_d,YWTC_d,YWTC_h);
          GPU_dbl_small_R_add_YWTC(rows,cols,size_tiles,k,A_d,YWTC_d,A_h);
       }
       */
    }
}
