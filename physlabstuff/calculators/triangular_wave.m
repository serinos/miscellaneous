% Oct 2019
% Triangular Standing Wave simulation, tinkered code of Askin Kocabas

L = 1;
z = linspace(-L,L,100);
f = zeros(1,100);

numberOfModes = 4;

for n = 1:2:2*numberOfModes-1
    mode(1,:) = (-0.81055)*sin(2*1*pi*z/(1*L));
    mode(3,:) = (0.090039)*sin(2*3*pi*z/(1*L));
    mode(5,:) = (-0.03237)*sin(2*5*pi*z/(1*L));
    mode(7,:) = (0.016536)*sin(2*7*pi*z/(1*L));
    f(1,:)=f(1,:)+mode(n,:);
end

plot(f(1,1:100));

for t = 0:0.01:2
    fT = zeros(1,100);
for n = 1:2:2*numberOfModes-1
    modeT(n,:) = mode(n,:)*cos((((n*pi/L)))*t);
    fT(1,:) = fT(1,:) + modeT(n,:);
end

plot(z,fT,'lineWidth',5);
hold on;
for n = 1:2:2*numberOfModes-1
    plot(z,modeT(n,:));
end
hold off;
axis([0 L -2 2]);
pause(0.1);
end
