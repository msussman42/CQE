      recursive subroutine permutations(k,n,nfact,counter,A,B)
      integer, intent(in) :: k,n,nfact
      integer, intent(inout) :: counter
      integer, intent(inout), dimension(n) :: A
      integer, intent(inout), dimension(nfact,n) :: B
      integer i,j,hold

      i=1
      do j=1,n
       i=i*j
      enddo
      if (i.eq.nfact) then
       !do nothing
      else
       print *,"i <> nfact"
       stop
      endif
      if ((counter.ge.1).and.(counter.le.nfact)) then
       !do nothing
      else
       print *,"counter invalid"
       stop
      endif

      if (k.eq.1) then
       do i=1,n
        B(counter,i)=A(i)
       enddo
       counter=counter+1
      else
       call permutations(k-1,n,nfact,counter,A,B)
       do i=0,k-2
        if (2*(k/2).eq.k) then
         hold=A(i+1)
         A(i+1)=A(k)
         A(k)=hold
        else
         hold=A(1)
         A(1)=A(k)
         A(k)=hold
        endif
        call permutations(k-1,n,nfact,counter,A,B)
       enddo
      endif

      return
      end subroutine permutations

      subroutine check_iso(nv,nvfact,ne,A,B,nv_permute)
      IMPLICIT NONE

      integer, intent(in) :: ne,nv,nvfact
      integer, intent(in), dimension(ne,nv) :: A,B
      integer, intent(in), dimension(nvfact,nv) :: nv_permute
      integer i,j,i1,j1,iso_flag,k,k2,i3
      real*8 counter,total_counter,sub_counter
      integer, dimension(ne,nv) :: Bhold

      print *,"nvfact= ",nvfact

      i=1
      do j=1,nv
       i=i*j
      enddo
      if (i.eq.nvfact) then
       !do nothing
      else
       print *,"i <> nvfact"
       stop
      endif

      counter=1.0d0
      sub_counter=1.0d0
      total_counter=nvfact
      do j=1,nvfact
       counter=counter+1.0d0
       sub_counter=sub_counter+1.0d0
       if (sub_counter.ge.100.0) then
        print *,"counter= ",counter
        print *,"testing j=",j
        print *,"total counter= ",total_counter
        sub_counter=1.0d0
       endif
       iso_flag=1
       do i1=1,ne
       do j1=1,nv
        Bhold(i1,j1)=B(i1,nv_permute(j,j1))
        if (Bhold(i1,j1).ne.A(i1,j1)) then
         iso_flag=0
        endif
        if ((Bhold(i1,j1).eq.0).or. &
            (Bhold(i1,j1).eq.1)) then
         !do nothing
        else
         print *,"i1,j1,Bhold invalid: ",i1,j1,Bhold(i1,j1)
         stop
        endif
        if ((A(i1,j1).eq.0).or. &
            (A(i1,j1).eq.1)) then
         !do nothing
        else
         print *,"i1,j1,A invalid: ",i1,j1,A(i1,j1)
         stop
        endif
       enddo
       enddo
       if (iso_flag.eq.1) then

        print *,"is_flag=1 for j=",j
        do j1=1,nv
         print *,"j1,nv_permute(j,j1) ",j1,nv_permute(j,j1)
        enddo
       endif
      enddo

      return
      end subroutine check_iso


      program main
      IMPLICIT NONE
      integer, parameter :: ne=11
      integer, parameter :: nv=6
      integer, parameter :: nvfact=720

      integer, dimension(ne,nv) :: A,B,C
      integer, dimension(nvfact,nv) :: nv_permute
      integer, dimension(nv) :: nv_single
      integer counter,i,j
      
      do i=1,nv
       nv_single(i)=i
      enddo
      counter=1
      call permutations(nv,nv,nvfact,counter,nv_single,nv_permute)
      do i=1,ne
      do j=1,nv
       A(i,j)=0 
       B(i,j)=0 
       C(i,j)=0 
      enddo
      enddo

      A(1,1)=1
      A(1,6)=1
      A(2,5)=1
      A(2,6)=1
      A(3,4)=1
      A(3,5)=1
      A(4,3)=1
      A(4,4)=1
      A(5,2)=1
      A(5,3)=1
      A(6,1)=1
      A(6,2)=1

      A(7,4)=1
      A(8,1)=1
      A(8,3)=1
      A(9,4)=1
      A(9,6)=1
      A(10,2)=1
      A(10,6)=1
      A(11,3)=1
      A(11,5)=1

      B(1,1)=1
      B(1,6)=1
      B(2,5)=1
      B(2,6)=1
      B(3,4)=1
      B(3,5)=1
      B(4,3)=1
      B(4,4)=1
      B(5,2)=1
      B(5,3)=1
      B(6,1)=1
      B(6,2)=1

      B(7,1)=1
      B(7,4)=1
      B(8,6)=1
      B(8,3)=1
      B(9,1)=1
      B(9,5)=1
      B(10,4)=1
      B(10,6)=1
      B(11,2)=1
      B(11,5)=1

      C(1,1)=1
      C(1,6)=1
      C(2,5)=1
      C(2,6)=1
      C(3,4)=1
      C(3,5)=1
      C(4,3)=1
      C(4,4)=1
      C(5,2)=1
      C(5,3)=1
      C(6,1)=1
      C(6,2)=1

      C(7,1)=1
      C(7,4)=1
      C(8,1)=1
      C(8,5)=1
      C(9,4)=1
      C(9,6)=1
      C(10,2)=1
      C(10,6)=1
      C(11,3)=1
      C(11,5)=1

      print *,"checking A and B"
      call check_iso(nv,nvfact,ne,A,B,nv_permute)

      print *,"checking B and C"
      call check_iso(nv,nvfact,ne,B,C,nv_permute)

      print *,"checking A and C"
      call check_iso(nv,nvfact,ne,A,C,nv_permute)

      return
      end

