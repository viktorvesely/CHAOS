function J2 = jet2(m)
%JET    Variant of HSV.
%   JET(M), a variant of HSV(M), is the colormap used with the
%   NCSA fluid jet image.
%   JET, by itself, is the same length as the current colormap.
%   Use COLORMAP(JET).
%
%   See also HSV, HOT, PINK, FLAG, COLORMAP, RGBPLOT.

%   C. B. Moler, 5-10-91, 8-19-92.
%   Copyright (c) 1984-98 by The MathWorks, Inc.
%   $Revision: 5.3 $  $Date: 1997/11/21 23:33:51 $

if nargin < 1, m = size(get(gcf,'colormap'),1); end    %m = 64
m = 60;
n = max(round(m/4),1);                                 %n = 16
x = (1:n)'/n;
y = (n/2:n)'/n;
e = ones(length(x),1);
r = [0*y; 0*e; x; e; flipud(y)];
g = [0*y; x; e; flipud(x); 0*y];
b = [y; e; flipud(x); 0*e; 0*y];
J2 = [r g b];
while size(J2,1) > m
   J2(1,:) = [];
   if size(J2,1) > m, J2(size(J2,1),:) = []; end
end

% change the last colors
J2(61,:) = [0 0 0]; % black (61)
J2(62,:) = [1 0 0]; % red   (62)
J2(63,:) = [0 1 0]; % green (63)
%J2(64,:) = [0 0 1]; % blue  (64)
J2(64,:) = [1 1 1]; % white  (64)