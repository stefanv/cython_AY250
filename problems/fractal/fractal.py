import matplotlib.pyplot as plt
import numpy as np

ITERATIONS = 10
DENSITY = 100 # warning: execution speed decreases with square of DENSITY

x_min, x_max = -2, 1
y_min, y_max = -1.5, 1.5

z = np.zeros((DENSITY, DENSITY), dtype=complex)

dx = (x_max - x_min) / float(DENSITY)
dy = (y_max - y_min) / float(DENSITY)

for x in range(DENSITY):
    for y in range(DENSITY):
        z[y, x] = x_min + x * dx + 1j * (y_min + y * dy)

fractal = np.zeros(z.shape, dtype=np.uint8) + 255

for n in range(ITERATIONS):
    print "Iteration %d" % n

    for x in range(DENSITY):
        for y in range(DENSITY):
            c = x_min + x * dx + \
                1j * (y_min + y * dy)

            # --- Uncomment to see different sets ---

            # Tricorn
            # z[y, x] = z[y, x].conjugate()

            # Burning ship
            #z[y, x] = abs(z[y, x].real) + 1j*abs(z[y, x].imag)

            # ---

            # Leave the lines below in place

            z[y, x] = z[y, x]**2 + c

            if (fractal[y, x] == 255) and (abs(z[y, x]) > 10):
                fractal[y, x] = 254 * n / float(ITERATIONS)

            if (abs(z[y, x]) > 10):
                z[y, x] = 0

plt.imshow(np.log(fractal), cmap=plt.cm.hot,
           extent=(x_min, x_max, y_min, y_max))
plt.title('Mandelbrot Set')
plt.xlabel('Re(z)')
plt.ylabel('Im(z)')
plt.show()
