% July 31 2019 - Onur Serin
% Converts csv output of oscillator to xlsx after dumping header, plots
% in new window, used for AC
% Filename: converted.xlsx; A--> Time, B --> Channel 1
% Calculates fwhm for each spike

clear
csv_handler = fopen('tek0000.csv');
csv_data = textscan(csv_handler,'%f%f%f','HeaderLines',18,'delimiter',',');
fseek(csv_handler,0,'bof');
delta_tau_data = textscan(csv_handler,'%s%f', 1,'HeaderLines',6,'delimiter',',');
delta_tau = delta_tau_data{2};
fclose(csv_handler);

delta_x = 40*10^-6;
c = 3 * 10^8;
delta_t = (2 * delta_x) / c;
scaling_factor = delta_t / delta_tau;

col1 = csv_data{1};
col1_result = col1 * scaling_factor;
col2 = csv_data{2};
col2_min = min(col2(:));
col2_max = max(col2(:));
col2_result = (col2 - col2_min) / (col2_max - col2_min) * 2 + 1;

M = [col1_result col2_result];
i = M(:);
data_at_2 = zeros(1,2);
counter = 1;

% Play with the following three vars to fix points if necessary:
x = 0;
y = 20;
limit_demarkation = 0.01;

for m = 10001:20000;
   if and(i(m + x) > 2 - limit_demarkation, i(m + x) < 2 + limit_demarkation);
           data_at_2(counter) = i(m -10000 + x);
           counter = counter + 1;
           x = x + y;
           i = [i; zeros(y,1)];
   end
end

display("FWHM (fs):")
for n = 1:length(data_at_2)/2
    fwhm = data_at_2(2*n) - data_at_2(2*n - 1);
    display(fwhm * 10^15);
    
end
filename = 'converted.xlsx';
writematrix(M, filename);

plot(col1_result, col2_result)
