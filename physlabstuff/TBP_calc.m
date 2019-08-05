% Aug 5 - O.S
% Calculates Time Bandwidth Product

clear
display("All results in SI")
c = 299792458;
% Interactive:
%cursor1 = 10^-3 * input("Cursor1 in ms: ");
%cursor2 = 10^-3 * input("Cursor2 in ms: ");
%delta_x = 10^-6 * input("Δx in μm: ");
%delta_l = 10^-9 * input("Δl in nm: ");
%l_sqr = 10^-18 * (input("l in nm: "))^2;

% Plugging-in:
cursor1 = 10^-3 * (2.5);
cursor2 = 10^-3 * (0);
delta_x = 10^-6 * (100);
delta_l = 10^-9 * (37.7);
l_sqr = 10^-18 * (1276)^2;
FWHM_mean = 10^-6 * (286);
% End of plugging-in

delta_t = (2 * delta_x) / c;
delta_cursor = abs(cursor1 - cursor2);

% Interactive FWHM mean calculation:
%FWHM_matrix = [];
%temp = 1;
%while temp ~= 0
%   temp = 10^-6 * input("FWHM value in μs (Type 0 to end data retrieval): ");
%   if temp ~= 0
%       FWHM_matrix = [FWHM_matrix temp];
%   end
%end

%FWHM_mean = 0;
%for i = 1:length(FWHM_matrix)
%    FWHM_mean = FWHM_mean + FWHM_matrix(i);
%end
%FWHM_mean = FWHM_mean / length(FWHM_matrix);


tau_p = (FWHM_mean * delta_t) / (delta_cursor * 1.55);
delta_nu = (c * delta_l) / l_sqr;
TBP = delta_nu * tau_p;

display(tau_p)
display(TBP)