      recursive subroutine permutations(k,n,nfact,counter,A,B)
      IMPLICIT NONE
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

      subroutine check_iso(nv,nvfact,A,B,nv_permute)
      IMPLICIT NONE

      integer, intent(in) :: nv,nvfact
      integer, intent(in), dimension(nv,nv) :: A,B
      integer, intent(in), dimension(nvfact,nv) :: nv_permute
      integer i,j,i1,j1,iso_flag,k,k2,i3,expect_flag
      integer counter,total_counter,sub_counter
      integer, dimension(nv,nv) :: Bhold

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

      counter=1
      sub_counter=1
      total_counter=nvfact
      do j=1,nvfact
       counter=counter+1
       sub_counter=sub_counter+1
       if (sub_counter.ge.10000) then
        print *,"counter= ",counter
        print *,"testing j=",j
        print *,"total counter= ",total_counter
        sub_counter=1
       endif

       iso_flag=1
       do i1=1,nv
       do j1=1,nv
        Bhold(i1,j1)=B(nv_permute(j,i1),nv_permute(j,j1))
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
      integer, parameter :: nv=7
      integer, parameter :: nvfact=5040

      integer, dimension(nv,nv) :: A,B
      integer, dimension(nvfact,nv) :: nv_permute
      integer, dimension(nv) :: nv_single
      integer counter,i,j
      
      do i=1,nv
       nv_single(i)=i
      enddo
      counter=1
      call permutations(nv,nv,nvfact,counter,nv_single,nv_permute)
      do i=1,nv
      do j=1,nv
       A(i,j)=0 
       B(i,j)=0 
      enddo
      enddo
      do i=1,nv
       A(i,i)=1
       B(i,i)=1
      enddo


      A(1,2)=1
      A(1,3)=1
      A(1,6)=1
      A(1,7)=1

      A(2,7)=1
      A(2,4)=1
      A(2,3)=1

      A(3,4)=1
      A(3,5)=1

      A(4,5)=1
      A(4,6)=1

      A(5,6)=1
      A(5,7)=1

      A(6,7)=1

      B(1,5)=1
      B(1,4)=1
      B(1,2)=1
      B(1,7)=1

      B(2,6)=1
      B(2,4)=1
      B(2,3)=1

      B(3,4)=1
      B(3,6)=1
      B(3,7)=1

      B(4,5)=1

      B(5,6)=1
      B(5,7)=1

      B(6,7)=1

      do i=1,nv
      do j=1,nv
       if (i.lt.j) then
        A(j,i)=A(i,j)
        B(j,i)=B(i,j)
       endif
      enddo
      enddo

      print *,"---------------------"
      print *,"checking A and B"
      call check_iso(nv,nvfact,A,B,nv_permute)
      print *,"---------------------"

      return
      end

