/* File:     lab5_2_1.c
 *
 * Purpose:  Use MPI to rewrite code (At the same time keep the original part of openmp parallel) -- heated_plate_openmp.c 
 *           change the part of 'while()'
 *           and use "MPI_Pack()/MPI_Unpack()" or "MPI_Type_create_struct()"
 *           the part use MPI_Pack()/MPI_Unpack() in 
 *              *pack unpack*
 *                ...
 *              **
 *           the part use MPI_Type_create_struct() in 
 *           *MPI_Type_create_struct*
 *               ...
 *            **
 *
 * Compile:  mpicc -o lab5_2_1 lab5_2_1.c -fopenmp
 *
 * Run:      mpiexec -np <num> ./lab5_2_1
 *           <num> is the number of process
 *
 * Input:    none
 * Output:   some information like  time and interations and error.
 */
# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <omp.h>
# include <mpi.h>

int main ( int argc, char *argv[] );

/******************************************************************************/

int main ( int argc, char *argv[] )
{
# define M 500
# define N 500

  double diff;
  double epsilon = 0.001;
// double epsilon = 0.04;
  int i;
  int iterations;
  int iterations_print;
  int j;
  double mean;
  double my_diff;
  double u[M][N];
  double w[M][N];
  double wtime;

  MPI_Init(NULL, NULL);
  int numprocess,rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&numprocess);
  int block = (M-1-1)/numprocess;

  if(rank == 0){
    printf ( "\n" );
    printf ( "HEATED_PLATE_OPENMP\n" );
    printf ( "  C/OpenMP version\n" );
    printf ( "  A program to solve for the steady state temperature distribution\n" );
    printf ( "  over a rectangular plate.\n" );
    printf ( "\n" );
    printf ( "  Spatial grid of %d by %d points.\n", M, N );
    printf ( "  The iteration will be repeated until the change is <= %e\n", epsilon ); 
    printf ( "  Number of processors available = %d\n", omp_get_num_procs ( ) );
    printf ( "  Number of threads =              %d\n", omp_get_max_threads ( ) );
    /*
      Set the boundary values, which don't change. 
    */
      mean = 0.0;

    #pragma omp parallel shared ( w ) private ( i, j )
      {
    #pragma omp for
        for ( i = 1; i < M - 1; i++ )
        {
          w[i][0] = 100.0;
        }
    #pragma omp for
        for ( i = 1; i < M - 1; i++ )
        {
          w[i][N-1] = 100.0;
        }
    #pragma omp for
        for ( j = 0; j < N; j++ )
        {
          w[M-1][j] = 100.0;
        }
    #pragma omp for
        for ( j = 0; j < N; j++ )
        {
          w[0][j] = 0.0;
        }
    /*
      Average the boundary values, to come up with a reasonable
      initial value for the interior.
    */
    #pragma omp for reduction ( + : mean )
        for ( i = 1; i < M - 1; i++ )
        {
          mean = mean + w[i][0] + w[i][N-1];
        }
    #pragma omp for reduction ( + : mean )
        for ( j = 0; j < N; j++ )
        {
          mean = mean + w[M-1][j] + w[0][j];
        }
      }
    /*
      OpenMP note:
      You cannot normalize MEAN inside the parallel region.  It
      only gets its correct value once you leave the parallel region.
      So we interrupt the parallel region, set MEAN, and go back in.
    */
      mean = mean / ( double ) ( 2 * M + 2 * N - 4 );
      printf ( "\n" );
      printf ( "  MEAN = %f\n", mean );
    /* 
      Initialize the interior solution to the mean value.
    */
    #pragma omp parallel shared ( mean, w ) private ( i, j )
      {
    #pragma omp for
        for ( i = 1; i < M - 1; i++ )
        {
          for ( j = 1; j < N - 1; j++ )
          {
            w[i][j] = mean;
          }
        }
      }
    /*
      iterate until the  new solution W differs from the old solution U
      by no more than EPSILON.
    */
    printf ( "\n" );
    printf ( " Iteration  Change\n" );
    printf ( "\n" );


    iterations = 0;
    iterations_print = 1;

    wtime = omp_get_wtime ( );
  }


  diff = epsilon;
  int first = 1;

  int start,end;
  start = 1+rank*block;
    if (rank==numprocess-1)
    {
      end =M-1;
    }
    else{
      end = 1+(rank+1)*block;
    }

    while ( epsilon <= diff )
  {
  /*
    Save the old solution in U.
  */
      # pragma omp parallel for private ( i, j ) shared ( u, w )
      for ( i = start; i < end; i++ ) 
      {
        for ( j = 0; j < N; j++ )
        {
          u[i][j] = w[i][j];//u[i-1],u[i+1] impact
        }
      }
/**pack unpack**/
    int position;
    double buffer[2*M*N];
/****/

    /**MPI_Type_create_struct**/
    struct var{
        double u[M*N];//u[M][N]
        double w[M*N];//w[M][N]
    };

    int var_count=2;
    int var_everycount[2]={M*N, M*N};
    MPI_Aint var_displace[2]={0, 8*M*N};
    MPI_Datatype var_type[2]={MPI_DOUBLE,MPI_DOUBLE};
    MPI_Datatype mytype;    

    MPI_Type_create_struct(var_count, var_everycount, var_displace, var_type, &mytype);
    MPI_Type_commit(&mytype);
    /****/

    if(first == 1 && rank==0){
      first = 0;
      # pragma omp parallel for private ( i, j ) shared ( u, w )
          for ( i = 0; i < M; i++ ) 
          {
            for ( j = 0; j < N; j++ )
            {
              u[i][j] = w[i][j];//u[i-1],u[i+1] impact
            }
          }

 /**MPI_Type_create_struct**/      
      struct var temp;
      for(int i=0;i<M;++i){
        for(int j=0;j<N;++j){
        temp.u[i*M+j]=u[i][j];//u[i][j]
        temp.w[i*M+j]=w[i][j];//w[i][j]
        }
      }
/****/

/**pack unpack**/
      // MPI_Pack(u, M*N, MPI_DOUBLE, buffer, 2*M*N*8, &position, MPI_COMM_WORLD);
      // MPI_Pack(w, M*N, MPI_DOUBLE, buffer, 2*M*N*8, &position, MPI_COMM_WORLD);
/****/

      for(int i=1;i<numprocess;++i){
        /**pack unpack**/
        // MPI_Send(buffer,2*M*N,MPI_DOUBLE,i,1,MPI_COMM_WORLD);
        /****/

        /**MPI_Type_create_struct**/
        MPI_Send(&temp,1,mytype,i,1,MPI_COMM_WORLD);
        /****/
      }
    }

    else if (first == 1 && rank!=0) {
      first = 0;
      /**MPI_Type_create_struct**/
      struct var temp;
      MPI_Recv(&temp,1,mytype,0,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      for(int i=0;i<M;++i){
        for(int j=0;j<N;++j){
          u[i][j]=temp.u[i*M+j];//u[i][j]
          w[i][j]=temp.w[i*M+j];//w[i][j]
        }
      }
      /****/

      /**pack unpack**/
      // MPI_Recv(buffer,2*M*N,MPI_DOUBLE,0,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      // MPI_Unpack(buffer, 2*M*N, &position, u, M*N, MPI_DOUBLE, MPI_COMM_WORLD);
      // MPI_Unpack(buffer, 2*M*N, &position, w, M*N, MPI_DOUBLE, MPI_COMM_WORLD);
      /****/
    }

    else{
      if(rank>0){
      MPI_Recv(u[start-1],N,MPI_DOUBLE,rank-1,3,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      MPI_Send(u[start],N,MPI_DOUBLE,rank-1,4,MPI_COMM_WORLD);}
      if(rank<numprocess-1){
      MPI_Send(u[end-1],N,MPI_DOUBLE,rank+1,3,MPI_COMM_WORLD);
      MPI_Recv(u[end],N,MPI_DOUBLE,rank+1,4,MPI_COMM_WORLD,MPI_STATUS_IGNORE);}
    }

      
    # pragma omp parallel for private ( i, j ) shared ( u, w )
    for ( i = start; i < end; i++ )
      {
        for ( j = 1; j < N - 1; j++ )
        {
          w[i][j] = ( u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1] ) / 4.0;
        }
    }


    diff = 0.0;
    my_diff = 0.0;

    for ( i = start; i < end; i++ )
    {
      for ( j = 1; j < N - 1; j++ )
      {
        if ( my_diff < fabs ( w[i][j] - u[i][j] ) )
        {
          my_diff = fabs ( w[i][j] - u[i][j] );
        }
      }
    }

    double all_diff[numprocess];
    MPI_Allgather(&my_diff,1,MPI_DOUBLE,all_diff,1,MPI_DOUBLE,MPI_COMM_WORLD);

    for(int i=0;i<numprocess;++i){
        if(diff < all_diff[i]){
            diff = all_diff[i];
        }
    }

    if(rank == 0){
      iterations++;
      if ( iterations == iterations_print )
      {
        printf ( "  %8d  %f\n", iterations, diff );
       iterations_print = 2 * iterations_print;
      }
    } 

  }

  if(rank == 0){
    wtime = omp_get_wtime ( ) - wtime;
    printf ( "\n" );
    printf ( "  %8d  %f\n", iterations, diff );
    printf ( "\n" );
    printf ( "  Error tolerance achieved.\n" );
    printf ( "  Wallclock time = %f\n", wtime );
  /*
    Terminate.
  */
    printf ( "\n" );
    printf ( "HEATED_PLATE_OPENMP:\n" );
    printf ( "  Normal end of execution.\n" );
  }

  MPI_Finalize();

  return 0;

  # undef M
  # undef N
}
