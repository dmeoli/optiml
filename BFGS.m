function [ x , status ] =  BFGS( f , varargin )

%function [ x , status ] = BFGS( f , x , delta , eps , MaxFeval , m1 , m2 ,
%                                tau , sfgrd , MInf , mina )
%
% Apply a Quasi-Newton approach, in particular using the celebrated
% Broyden-Fletcher-Goldfarb-Shanno (BFGS) formula, for the minimization of
% the provided function f, which must have the following interface:
%
%   [ v , g ] = f( x )
%
% Input:
%
% - x is either a [ n x 1 ] real (column) vector denoting the input of
%   f(), or [] (empty).
%
% Output:
%
% - v (real, scalar): if x == [] this is the best known lower bound on
%   the unconstrained global optimum of f(); it can be -Inf if either f()
%   is not bounded below, or no such information is available. If x ~= []
%   then v = f(x).
%
% - g (real, [ n x 1 ] real vector): this also depends on x. if x == []
%   this is the standard starting point from which the algorithm should
%   start, otherwise it is the gradient of f() at x (or a subgradient if
%   f() is not differentiable at x, which it should not be if you are
%   applying the gradient method to it).
%
% The other [optional] input parameters are:
%
% - x (either [ n x 1 ] real vector or [], default []): starting point.
%   If x == [], the default starting point provided by f() is used.
%
% - delta (real scalar, optional, default value 1): the initial
%   approximation of the Hesssian is taken as delta * I if delta > 0;
%   otherwise, the initial Hessian is approximated by finite differences
%   with - delta as the step, and inverted just the once.
%
% - eps (real scalar, optional, default value 1e-6): the accuracy in the
%   stopping criterion: the algorithm is stopped when the norm of the
%   gradient is less than or equal to eps. If a negative value is provided,
%   this is used in a *relative* stopping criterion: the algorithm is
%   stopped when the norm of the gradient is less than or equal to
%   (- eps) * || norm of the first gradient ||.
%
% - MaxFeval (integer scalar, optional, default value 1000): the maximum
%   number of function evaluations (hence, iterations will be not more than
%   MaxFeval because at each iteration at least a function evaluation is
%   performed, possibly more due to the line search).
%
% - m1 (real scalar, optional, default value 0.01): first parameter of the
%   Armijo-Wolfe-type line search (sufficient decrease). Has to be in (0,1)
%
% - m2 (real scalar, optional, default value 0.9): typically the second
%   parameter of the Armijo-Wolfe-type line search (strong curvature
%   condition). It should to be in (0,1); if not, it is taken to mean that
%   the simpler Backtracking line search should be used instead
%
% - tau (real scalar, optional, default value 0.9): scaling parameter for
%   the line search. In the Armijo-Wolfe line search it is used in the
%   first phase: if the derivative is not positive, then the step is
%   divided by tau (which is < 1, hence it is increased). In the
%   Backtracking line search, each time the step is multiplied by tau
%   (hence it is decreased).
%
% - sfgrd (real scalar, optional, default value 0.01): safeguard parameter
%   for the line search. to avoid numerical problems that can occur with
%   the quadratic interpolation if the derivative at one endpoint is too
%   large w.r.t. the one at the other (which leads to choosing a point
%   extremely near to the other endpoint), a *safeguarded* version of
%   interpolation is used whereby the new point is chosen in the interval
%   [ as * ( 1 + sfgrd ) , am * ( 1 - sfgrd ) ], being [ as , am ] the
%   current interval, whatever quadratic interpolation says. If you
%   experiemce problems with the line search taking too many iterations to
%   converge at "nasty" points, try to increase this
%
% - MInf (real scalar, optional, default value -Inf): if the algorithm
%   determines a value for f() <= MInf this is taken as an indication that
%   the problem is unbounded below and computation is stopped
%   (a "finite -Inf").
%
% - mina (real scalar, optional, default value 1e-16): if the algorithm
%   determines a stepsize value <= mina, this is taken as an indication
%   that something has gone wrong (the gradient is not a direction of
%   descent, so maybe the function is not differentiable) and computation
%   is stopped. It is legal to take mina = 0, thereby in fact skipping this
%   test.
%
% Output:
%
% - x ([ n x 1 ] real column vector): the best solution found so far.
%
% - status (string): a string describing the status of the algorithm at
%   termination
%
%   = 'optimal': the algorithm terminated having proven that x is a(n
%     approximately) optimal solution, i.e., the norm of the gradient at x
%     is less than the required threshold
%
%   = 'unbounded': the algorithm has determined an extrenely large negative
%     value for f() that is taken as an indication that the problem is
%     unbounded below (a "finite -Inf", see MInf above)
%
%   = 'stopped': the algorithm terminated having exhausted the maximum
%     number of iterations: x is the bast solution found so far, but not
%     necessarily the optimal one
%
%   = 'error': the algorithm found a numerical error that prevents it from
%     continuing optimization (see mina above)
%
%{
 =======================================
 Author: Antonio Frangioni
 Date: 10-11-17
 Version 1.10
 Copyright Antonio Frangioni
 =======================================
%}

Plotf = true;  % if f and the trajectory have to be plotted when n = 2

% reading and checking input- - - - - - - - - - - - - - - - - - - - - - - -
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if ~ isa( f , 'function_handle' )
   error( 'f not a function' );
end

if isempty( varargin ) || isempty( varargin{ 1 } )
   [ fStar , x ] = f( [] );
else
   x = varargin{ 1 };
   if ~ isreal( x )
      error( 'x not a real vector' );
   end

   if size( x , 2 ) ~= 1
      error( 'x is not a (column) vector' );
   end

   fStar = f( [] );
end

n = size( x , 1 );

if length( varargin ) > 1
   delta = varargin{ 2 };
   if ~ isscalar( delta )
      error( 'delta is not a real scalar' );
   end
else
   delta = 1;
end

if length( varargin ) > 2
   eps = varargin{ 3 };
   if ~ isreal( eps ) || ~ isscalar( eps )
      error( 'eps is not a real scalar' );
   end
else
   eps = 1e-6;
end

if length( varargin ) > 3
   MaxFeval = round( varargin{ 4 } );
   if ~ isscalar( MaxFeval )
      error( 'MaxFeval is not an integer scalar' );
   end
else
   MaxFeval = 1000;
end

if length( varargin ) > 4
   m1 = varargin{ 5 };
   if ~ isscalar( m1 )
      error( 'm1 is not a real scalar' );
   end
   if m1 <= 0 || m1 >= 1
      error( 'm1 is not in (0 ,1)' );
   end
else
   m1 = 0.01;
end

if length( varargin ) > 5
   m2 = varargin{ 6 };
   if ~ isscalar( m1 )
      error( 'm2 is not a real scalar' );
   end
else
   m2 = 0.9;
end

AWLS = ( m2 > 0 && m2 < 1 );

if length( varargin ) > 6
   tau = varargin{ 7 };
   if ~ isscalar( tau )
      error( 'tau is not a real scalar' );
   end
   if tau <= 0 || tau >= 1
      error( 'tau is not in (0 ,1)' );
   end
else
   tau = 0.9;
end

if length( varargin ) > 7
   sfgrd = varargin{ 8 };
   if ~ isscalar( sfgrd )
      error( 'sfgrd is not a real scalar' );
   end
   if sfgrd <= 0 || sfgrd >= 1
      error( 'sfgrd is not in (0, 1)' );
   end
else
   sfgrd = 0.01;
end

if length( varargin ) > 8
   MInf = varargin{ 9 };
   if ~ isscalar( MInf )
      error( 'MInf is not a real scalar' );
   end
else
   MInf = - Inf;
end

if length( varargin ) > 9
   mina = varargin{ 10 };
   if ~ isscalar( mina )
      error( 'mina is not a real scalar' );
   end
   if mina < 0
      error( 'mina is < 0' );
   end
else
   mina = 1e-16;
end

% "global" variables- - - - - - - - - - - - - - - - - - - - - - - - - - - -
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

lastx = zeros( n , 1 );  % last point visited in the line search
lastg = zeros( n , 1 );  % gradient of lastx
feval = 1;               % f() evaluations count ("common" with LSs)

% initializations - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

fprintf( 'BFGS method\n');
if fStar > - Inf
   fprintf( 'feval\trel gap');
else
   fprintf( 'feval\tf(x)');
end
fprintf( '\t\t|| g(x) ||\tls fev\ta*\t rho\n\n');

[ v , g ] = f( x );
ng = norm( g );
if eps < 0
   ng0 = - ng;  % norm of first subgradient: why is there a "-"? ;-)
else
   ng0 = 1;     % un-scaled stopping criterion
end

if delta > 0
   % initial approximation of inverse of Hessian = scaled identity
   B = delta * eye( n );
else
   % initial approximation of inverse of Hessian computed by finite
   % differences of gradient
   smallsetp = max( [ - delta , 1e-8 ] );
   B = zeros( n , n );
   for i = 1 : n
       xp = x;
       xp( i ) = xp( i ) + smallsetp;
       [ ~ , gp ] = f( xp );
       B( i , : ) = ( gp - g ) ./ smallsetp;
   end
   B = ( B + B' ) / 2;   % ensure it is symmetric
   lambdan = eigs( B , 1 , 'sa' );  % smallest eigenvalue
   if lambdan < 1e-6
      B = B + ( 1e-6 - lambdan ) * eye( n );
   end
   B = inv( B );
end

% main loop - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

while true

    % output statistics - - - - - - - - - - - - - - - - - - - - - - - - - -

    if fStar > - Inf
       fprintf( '%4d\t%1.4e\t%1.4e' , feval , ...
                ( v - fStar ) / max( [ abs( fStar ) 1 ] ) , ng );
    else
       fprintf( '%4d\t%1.8e\t\t%1.4e' , feval , v , ng );
    end

    % stopping criteria - - - - - - - - - - - - - - - - - - - - - - - - - -

    if ng <= eps * ng0;
       status = 'optimal';
       break;
    end

    if feval > MaxFeval
       status = 'stopped';
       break;
    end

    % compute approximation to Newton's direction - - - - - - - - - - - - -

    d = - B * g;

    % compute step size - - - - - - - - - - - - - - - - - - - - - - - - - -
    % as in Newton's method, the default initial stepsize is 1

    phip0 = g' * d;

    if AWLS
       [ a , v ] = ArmijoWolfeLS( v , phip0 , 1 , m1 , m2 , tau );
    else
       [ a , v ] = BacktrackingLS( v , phip0 , 1 , m1 , tau );
    end

    % output statistics - - - - - - - - - - - - - - - - - - - - - - - - - -

    fprintf( '\t%1.2e' , a );

    if a <= mina
       status = 'error';
       break;
    end

    if v <= MInf
       status = 'unbounded';
       break;
    end

    % update approximation of the Hessian - - - - - - - - - - - - - - - - -
    % warning: magic at work! Broyden-Fletcher-Goldfarb-Shanno formula

    s = lastx - x;   % s^i = x^{i + 1} - x^i
    y = lastg - g;   % y^i = \nabla f( x^{i + 1} ) - \nabla f( x^i )

    rho = y' * s;
    if rho < 1e-16
       fprintf( '\nError: y^i s^i = %1.2e\n' , rho );
       status = 'error';
       break;
    end

    rho = 1 / rho;

    fprintf( ' %1.2e\n' , rho );

    D = B * y * s';
    B = B + rho * ( ( 1 + rho * y' * B * y ) * ( s * s' ) - D - D' );

    % compute new point - - - - - - - - - - - - - - - - - - - - - - - - - -

    % possibly plot the trajectory
    if n == 2 && Plotf
       PXY = [ x ,  lastx ];
       line( 'XData' , PXY( 1 , : ) , 'YData' , PXY( 2 , : ) , ...
             'LineStyle' , '-' , 'LineWidth' , 2 ,  'Marker' , 'o' , ...
             'Color' , [ 0 0 0 ] );
       pause;
    end

    x = lastx;

    % update gradient - - - - - - - - - - - - - - - - - - - - - - - - - - -

    g = lastg;
    ng = norm( g );

    % iterate - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

end

% end of main loop- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
% inner functions - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

function [ phi , phip ] = f2phi( alpha )
% phi( alpha ) = f( x + alpha * d )
% phi'( alpha ) = < \nabla f( x + alpha * d ) , d >

   lastx = x + alpha * d;
   [ phi , lastg ] = f( lastx );
   phip = d' * lastg;
   feval = feval + 1;
end

% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

function [ a , phia ] = ArmijoWolfeLS( phi0 , phip0 , as , m1 , m2 , tau )

% performs an Armijo-Wolfe Line Search.
%
% phi0 = phi( 0 ), phip0 = phi'( 0 ) < 0
%
% as > 0 is the first value to be tested: if phi'( as ) < 0 then as is
% divided by tau < 1 (hence it is increased) until this does not happen
% any longer
%
% m1 and m2 are the standard Armijo-Wolfe parameters; note that the strong
% Wolfe condition is used
%
% returns the optimal step and the optimal f-value

lsiter = 1;  % count iterations of first phase
while feval <= MaxFeval
   [ phia , phips ] = f2phi( as );

   if ( phia <= phi0 + m1 * as * phip0 ) && ...
      ( abs( phips ) <= - m2 * phip0 )
      fprintf( '\t%2d' , lsiter );
      a = as;
      return;  % Armijo + strong Wolfe satisfied, we are done

   end
   if phips >= 0  % derivative is positive, break
      break;
   end
   as = as / tau;
   lsiter = lsiter + 1;
end

fprintf( '\t%2d ' , lsiter );
lsiter = 1;  % count iterations of second phase

am = 0;
a = as;
phipm = phip0;
while ( feval <= MaxFeval ) && ( ( as - am ) ) > mina && ( phips > 1e-12 )

   % compute the new value by safeguarded quadratic interpolation
   a = ( am * phips - as * phipm ) / ( phips - phipm );
   a = max( [ am * ( 1 + sfgrd ) min( [ as * ( 1 - sfgrd ) a ] ) ] );

   % compute phi( a )
   [ phia , phip ] = f2phi( a );

   if ( phia <= phi0 + m1 * a * phip0 ) && ( abs( phip ) <= - m2 * phip0 )
      break;  % Armijo + strong Wolfe satisfied, we are done
   end

   % restrict the interval based on sign of the derivative in a
   if phip < 0
      am = a;
      phipm = phip;
   else
      as = a;
      if as <= mina
         break;
      end
      phips = phip;
   end
   lsiter = lsiter + 1;
end

fprintf( '%2d' , lsiter );

end

% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

function [ as , phia ] = BacktrackingLS( phi0 , phip0 , as , m1 , tau )

% performs a Backtracking Line Search.
%
% phi0 = phi( 0 ), phip0 = phi'( 0 ) < 0
%
% as > 0 is the first value to be tested, which is decreased by
% multiplying it by tau < 1 until the Armijo condition with parameter
% m1 is satisfied
%
% returns the optimal step and the optimal f-value

lsiter = 1;  % count ls iterations
while feval <= MaxFeval && as > mina
   [ phia , ~ ] = f2phi( as );
   if phia <= phi0 + m1 * as * phip0  % Armijo satisfied
      break;                          % we are done
   end
   as = as * tau;
   lsiter = lsiter + 1;
end

fprintf( '\t%2d' , lsiter );

end

% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

end  % the end- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -