import numpy as np

thrust_kgf = 5




k_thrust_kgf=1.70105637e-8

k_moment=8.43409083e-11


kt_N=k_thrust_kgf * 9.8066  # Convert kgf to N

max_rpm=17144
max_rads = max_rpm * (2 * np.pi / 60)  # Convert RPM to radians per second
# Calculate thrust in Newtons
thrust_from_meas = thrust_kgf * 9.8066  # Convert kgf to N
thrust_from_kt = kt_N * max_rpm**2
kt_rads = thrust_from_kt / (max_rads**2)
thrust_from_rads = kt_rads * max_rads**2


#Calculate moment in Newton-meters per rad/s
moment_from_km=k_moment * (max_rpm**2)


print(f"Moment from k_moment: {moment_from_km} N*m/(rad/s)^2")



# print(f"Thrust from measurement: {thrust_from_meas:.2f} N")
# print(f"Thrust from kt: {thrust_from_kt:.2f} N")
# print(f"Thrust from rads: {thrust_from_rads:.2f} N")
# print(f"kt_rads: {kt_rads:.2e} N/(rad/s)^2")

# print (f"Thrust: {thrust:.2f} kgf")
#print(f"Motor RPM: {motor_rpm:.2f}")