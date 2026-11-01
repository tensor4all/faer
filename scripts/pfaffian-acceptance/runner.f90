program bench_pf
  use iso_fortran_env, only: real64, int64
  use ieee_arithmetic, only: ieee_is_finite
  use iso_c_binding, only: c_int
  implicit none
  interface
    subroutine set_blas_threads(t) bind(C,name='openblas_set_num_threads')
      import c_int
      integer(c_int), value :: t
    end subroutine
    function blas_threads() bind(C,name='openblas_get_num_threads') result(t)
      import c_int
      integer(c_int) :: t
    end function
  end interface
  character(2048) :: path, arg, dtype, smoke_arg
  integer :: n, unit, info, lw, rep, i, j, width, actual_threads, warmups
  integer, allocatable :: iw(:)
  integer(int64) :: t0, t1, rate, prime_start, iterations(10)
  real(real64), allocatable :: ar(:,:), ref_r(:,:), wr(:), rw(:)
  complex(real64), allocatable :: az(:,:), ref_z(:,:), wz(:)
  real(real64) :: p, q(1), samples(10), sorted(10), temp, median, pr, pi
  complex(real64) :: zp, zq(1)
  logical :: smoke
  call get_command_argument(1,path)
  call get_command_argument(2,arg); read(arg,*) n
  call get_command_argument(3,dtype)
  if(command_argument_count()/=4.and.command_argument_count()/=5)stop 16
  call get_command_argument(4,arg); read(arg,*) width
  if(width/=1)stop 16
  smoke=.false.
  if(command_argument_count()==5)then
    call get_command_argument(5,smoke_arg)
    if(trim(smoke_arg)/='--smoke')stop 16
    smoke=.true.
  endif
  call set_blas_threads(int(width,c_int))
  actual_threads=blas_threads()
  if(actual_threads/=width)stop 15
  allocate(iw(n),rw(max(1,n-1)))
  samples=0; iterations=0; pr=0; pi=0
  if(trim(dtype)=='f64')then
    allocate(ar(n,n),ref_r(n,n))
    open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='old',convert='little_endian')
    read(unit)ref_r;close(unit)
    ar=ref_r
    call dskpfa('L','P',n,ar,n,p,iw,q,-1,info)
    if(info/=0)stop 10
    lw=max(1,int(q(1)));allocate(wr(lw))
  else
    if(trim(dtype)/='c64')stop 11
    allocate(az(n,n),ref_z(n,n))
    open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='old',convert='little_endian')
    read(unit)ref_z;close(unit)
    az=ref_z
    call zskpfa('L','P',n,az,n,zp,iw,zq,-1,rw,info)
    if(info/=0)stop 12
    lw=max(1,int(real(zq(1),real64)));allocate(wz(lw))
  endif
  call system_clock(count_rate=rate)
  if(smoke)then
    call run_once()
  else
    call system_clock(prime_start)
    warmups=0
    do
      call run_once()
      warmups=warmups+1
      call system_clock(t1)
      if(warmups>=3.and.real(t1-prime_start,real64)>=0.5_real64*real(rate,real64))exit
    enddo
    do rep=1,10
      call system_clock(t0)
      do
        call run_once()
        iterations(rep)=iterations(rep)+1
        call system_clock(t1)
        if(real(t1-t0,real64)>=0.5_real64*real(rate,real64))exit
      enddo
      samples(rep)=real(t1-t0,real64)/real(rate,real64)/real(iterations(rep),real64)
    enddo
  endif
  if(trim(dtype)=='f64')then
    pr=p
  else
    pr=real(zp,real64);pi=aimag(zp)
  endif
  if(.not.ieee_is_finite(pr).or..not.ieee_is_finite(pi))stop 14
  sorted=samples
  do i=2,10
    temp=sorted(i);j=i-1
    do while(j>=1)
      if(sorted(j)<=temp)exit
      sorted(j+1)=sorted(j);j=j-1
    enddo
    sorted(j+1)=temp
  enddo
  median=(sorted(5)+sorted(6))/2
  write(*,'(A,I0,A,A,A,I0,A,I0,A,ES24.16E3,A,ES24.16E3,A,ES24.16E3,A)',advance='no') &
    '{"implementation":"fortran-pfapack","n":',n,',"dtype":"',trim(dtype), &
    '","threads":',width,',"blas_threads":',actual_threads,',"median_seconds":',median,',"pf_re":',pr,',"pf_im":',pi,',"samples_seconds":['
  do i=1,10
    if(i>1)write(*,'(A)',advance='no')','
    write(*,'(ES24.16E3)',advance='no')samples(i)
  enddo
  write(*,'(A)',advance='no')'],"batch_iterations":['
  do i=1,10
    if(i>1)write(*,'(A)',advance='no')','
    write(*,'(I0)',advance='no')iterations(i)
  enddo
  write(*,'(A)')'],"copy_included":true,"allocation_note":"caller matrices and optimal queried workspace preallocated; copy timed; library internal allocation not separately removed"}'
contains
  subroutine run_once()
    if(trim(dtype)=='f64')then
      ar=ref_r
      call dskpfa('L','P',n,ar,n,p,iw,wr,lw,info)
    else
      az=ref_z
      call zskpfa('L','P',n,az,n,zp,iw,wz,lw,rw,info)
    endif
    if(info/=0)stop 13
  end subroutine
end program
