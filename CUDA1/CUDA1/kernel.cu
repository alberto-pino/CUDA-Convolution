

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "lodepng.h"
#include <iostream>
#include <vector>
#include <string>

//int smBytes = (blockSize.x + 2 * R)*(blockSize.y + 2 * R) * sizeof(float) + K*K * sizeof(float);
//dim3 gridSize(((int)(H / blockSize.x)) + auxx, ((int)(L / blockSize.y)) + auxy);
__global__ void kernel(float *o_foto, float* i_foto, float* matriz, int K, int R, int dimx, int dimy, int dimx2, int dimy2)
{
	extern __shared__ float memory[];

	int blockDimy2 = (blockDim.y + R + R);//tamaño en "y" de los datos con halo que cada bloque debe procesar


	int global_ix = blockIdx.x*blockDim.x + threadIdx.x + R;//identificador global del hilo en x (+R es para salir del marco)
	int global_iy = blockIdx.y*blockDim.y + threadIdx.y + R;//identificador global del hilo en y (+R es para salir del marco)
	int global_i = global_ix*(dimy + R + R) + global_iy;//identificador global del hilo en región (+R+R es para salir del marco)

	int local_ix = threadIdx.x + R;//identificador local del hilo en x (+R es para salir del halo) 
	int local_iy = threadIdx.y + R;//identificador local del hilo en x (+R es para salir del halo) 
	int local_i = local_ix*(blockDimy2)+local_iy;


	int lMatriz_i = threadIdx.x*K + threadIdx.y + (blockDim.x + 2 * R)*(blockDim.y + 2 * R);//identificador local del hilo en la matriz

	bool t1 = ((global_ix + blockDim.x)*dimy2 + (global_iy) <dimx2*dimy2);//Condición para no salirse del array
	bool t2 = (((global_ix)*dimy2 + (global_iy + blockDim.y)) < dimx2*dimy2);//Condición para no salirse del array
	bool t3 = global_ix < (dimx2) && global_iy < (dimy2);//Condición para no salirse del array

														 //copiamos a memoria local la foto
	if (t3) //condición necesaria para no salirse del array
	{
		memory[local_i] = i_foto[global_i];//centro

		if ((threadIdx.x < R)) {

			memory[(local_ix - R)*blockDimy2 + (local_iy)] = i_foto[(global_ix - R)*dimy2 + (global_iy)];//arriba


			if (t1)
				memory[(local_ix + blockDim.x)*blockDimy2 + (local_iy)] = i_foto[(global_ix + blockDim.x)*dimy2 + (global_iy)];//abajo


			if (threadIdx.y < R)//para añadir esquinas
			{
				memory[(local_ix - R)*blockDimy2 + (local_iy - R)] = i_foto[(global_ix - R)*dimy2 + (global_iy - R)];//izq.arr
				if (t1)
					memory[(local_ix + blockDim.x)*blockDimy2 + (local_iy - R)] = i_foto[(global_ix + blockDim.x)*dimy2 + (global_iy - R)];//izq.abj
				if (t2)
					memory[(local_ix - R)*blockDimy2 + (local_iy + blockDim.y)] = i_foto[(global_ix - R)*dimy2 + (global_iy + blockDim.y)];//drc.arr
				if (t1&&t2)
					memory[(local_ix + blockDim.x)*blockDimy2 + (local_iy + blockDim.y)] = i_foto[(global_ix + blockDim.x)*dimy2 + (global_iy + blockDim.y)];//der.abj
			}
		}
		if ((threadIdx.y < R)) {//primera condición para añadir lados, segunda para no salirse del array

			memory[(local_ix)*blockDimy2 + (local_iy - R)] = i_foto[(global_ix)*dimy2 + (global_iy - R)];//izquierda
			if (t2)
				memory[(local_ix)*blockDimy2 + (local_iy + blockDim.y)] = i_foto[(global_ix)*dimy2 + (global_iy + blockDim.y)];//derecha		
		}
	}


	//copiamos a memoria local la matriz (Solo cuando los bloques tienen tamaño suficiente)
	if ((blockDim.x >= K) && (blockDim.y >= K)) {
		if ((threadIdx.x < K) && (threadIdx.y < K)) {
			memory[lMatriz_i] = matriz[threadIdx.x*K + threadIdx.y];
		}
	}

	//en caso contrario se leerá directamente de la matriz global.

	__syncthreads();

	float value = 0;
	int offsetx;
	int offsety;

	if ((blockDim.x >= K) && (blockDim.y >= K)) {
		if (global_ix < (dimx + R) && global_iy < (dimy + R)) {//solo queremos calcularlo para los elementos de la foto. Recordemos que el global ix ya está desplazado
			for (offsetx = -R; offsetx <= R; offsetx++) {
				for (offsety = -R; offsety <= R; offsety++) {
					value += memory[(local_ix + offsetx)*(blockDim.y + 2 * R) + local_iy + offsety] * memory[(offsetx + R)*K + R + offsety + (blockDim.x + 2 * R)*(blockDim.y + 2 * R)];
				}
			}
		}
	}
	else {
		if (global_ix < (dimx + R) && global_iy < (dimy + R)) {//solo queremos calcularlo para los elementos de la foto. Recordemos que el global ix ya está desplazado
			for (offsetx = -R; offsetx <= R; offsetx++) {
				for (offsety = -R; offsety <= R; offsety++) {
					value += memory[(local_ix + offsetx)*(blockDim.y + 2 * R) + local_iy + offsety] * matriz[(offsetx + R)*(K)+R + offsety];
				}
			}
		}
	}
	//__syncthreads();

	//copiamos a matriz o_foto
	if (global_ix < (dimx + R) && global_iy < (dimy + R)) {
		o_foto[(blockIdx.x*blockDim.x + threadIdx.x)*(dimy)+blockIdx.y*blockDim.y + threadIdx.y] = value;
	}
	__syncthreads();
}


void pasarHalo(int H, int L, int K, int R, int numFilas, int numColumnas, float* h_fotoPre, float* h_foto, std::vector<unsigned char> &image, int canal) {
	int i, j;

	//Cargamos foto (En este caso, nos inventamos valores);
	//interfaz de paso de foto como vector de unsigned char en 3 canales (RGB) a array blanco negro
	for (i = 0; i<H; i++) {
		for (j = 0; j<L; j++) {
			h_fotoPre[i*L + j] = (float)(image[4 * L * i + 4 * j + canal]);
		}
	}

	//Pasamos valores de foto a foto ampliada con el espacio del halo, que recordemos es un marco de grosor R

	for (i = 0; i<H; i++) {
		for (j = 0; j<L; j++) {
			h_foto[(i + R)*numColumnas + (j + R)] = h_fotoPre[i*L + j];
		}
	}

	//El primer paso del proceso consiste en dar valores al halo externo.
	//Aunque esto puede hacerse en la GPU, Dado que solo necesitamos hacerlo una vez, vamos a hacerlo en CPU
	int aux1 = 2 * R;
	int aux2H = 2 * (numFilas - R) - 2;
	int aux2L = 2 * (numColumnas - R) - 2;

	for (j = R; j < numColumnas - R; j++)
	{
		for (i = 0; i < R; i++)
		{
			h_foto[i*numColumnas + j] = h_foto[(aux1 - i)*numColumnas + j];
		}
		for (i = numFilas - R; i < numFilas; i++)
		{
			h_foto[i*numColumnas + j] = h_foto[(aux2H - i)*numColumnas + j];
		}
	}
	for (i = R; i < numFilas - R; i++)
	{
		for (j = 0; j < R; j++)
		{
			h_foto[i*numColumnas + j] = h_foto[i*numColumnas + aux1 - j];
		}
		for (j = numColumnas - R; j < numColumnas; j++)
		{
			h_foto[i*numColumnas + j] = h_foto[i*numColumnas + aux2L - j];
		}
	}
	for (i = 0; i < R; i++)
	{
		for (j = 0; j < R; j++)
		{
			h_foto[i*numColumnas + j] = h_foto[(aux1 - i)*numColumnas + aux1 - j];
		}
		for (j = numColumnas - R; j < numColumnas; j++)
		{
			h_foto[i*numColumnas + j] = h_foto[(aux1 - i)*numColumnas + aux2L - j];
		}
	}
	for (i = numFilas - R; i < numFilas; i++)
	{
		for (j = 0; j < R; j++)
		{
			h_foto[i*numColumnas + j] = h_foto[(aux2H - i)*numColumnas + aux1 - j];
		}
		for (j = numColumnas - R; j < numColumnas; j++)
		{
			h_foto[i*numColumnas + j] = h_foto[(aux2H - i)*numColumnas + aux2L - j];
		}
	}


}


int main()
{
	//cargamos imagen input

	const char* filename = "Dados.png";
	const char* filename_out = "Dados_out.png";

	std::vector<unsigned char> image; //the raw pixels
	std::vector<unsigned char> image_out; //the raw pixels
	unsigned width, height;

	//decode
	unsigned error = lodepng::decode(image, width, height, filename);

	//if there's an error, display it
	if (error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
	error = lodepng::decode(image_out, width, height, filename);

	if (error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;

	//Variables que vamos a usar:
	//iteradores
	int i, j;
	//pixels de la foto  (en nº filas/alto, nº columnas/ancho)
	int H, L;
	//Tamaño del filtro: (es cuadrado, de dimensiones kxk) y su radio R asociado
	int K, R;
	//Dimensiones del array que va a contener la foto aumentada.
	int numFilas, numColumnas;
	//puntero al array bidimensional de la foto, al array ampliado con el halo exterior y a la foto output
	float* h_fotoPre;
	float* h_foto;
	float* h_foto_out;
	//puntero al array que contiene la matriz kxk
	float* h_matriz;
	//punteros para la foto input en el device
	float *d_foto_in;
	//punteros para la matriz en device
	float * d_matriz;
	//punteros para foro output en device
	float * d_foto_out;
	//ints para numero de bloques y tamaño de bloques

	///Leemos matriz

	int c;
	int K1 = 1;
	bool checkM = true;
	int K2 = 1;
	FILE *matrix_in;
	matrix_in = fopen("matrix_in.txt", "r");

	if (matrix_in) {
		while ((c = getc(matrix_in)) != EOF) {
			if ((char)c == ';') {
				 K1 += 1;
				checkM = false;
			}
			if ((char)c == ','&& checkM) {
				if (checkM) K2 += 1;
			}
		}
		rewind(matrix_in);
	}

	//Inicializamos variables
	H = (int)height;
	L = (int)width;
	if (K1==K2) K = K1;
	R = (int)(K / 2);
	numFilas = H + 2 * R;
	numColumnas = L + 2 * R;


	//Reservamos memoria en device
	cudaMalloc((void**)&d_foto_in, sizeof(float)*numFilas*numColumnas);

	cudaMalloc((void**)&d_matriz, sizeof(float)*K*K);

	cudaMalloc((void**)&d_foto_out, sizeof(float)*H*L);


	//Reservamos memoria para foto, foto_aumentada, foto_out y filtro en host:
	h_fotoPre = (float*)malloc(sizeof(float)*H*L);

	h_foto = (float*)malloc(sizeof(float)*numFilas*numColumnas);

	h_foto_out = (float*)malloc(sizeof(float)*H*L);

	h_matriz = (float*)malloc(sizeof(float)*K*K);

	//Obtenemos tamaño de bloque y numero de bloques

	int aux = ((R / 16) + 1) * 16;
	dim3 blockSize(aux, aux);

	int auxx = 0; int auxy = 0;
	if (H % (2 * K) != 0) auxx = 1;
	if (L % (2 * K) != 0) auxy = 1;
	dim3 gridSize(((int)(H / blockSize.x)) + auxx, ((int)(L / blockSize.y)) + auxy);
	int smBytes = (blockSize.x + 2 * R)*(blockSize.y + 2 * R) * sizeof(float) + K*K * sizeof(float);//la memoria privada debe almacenar la matriz, y un bloque de foto		


	int canal_in;//variable para el canal de la foto
	int canal_out;

	///Creamos la matriz de convolución

	std::string num;
	int ifila = 0;
	int icolumna = 0;

	if (matrix_in) {
		while ((c = getc(matrix_in)) != EOF) {
			if (c==45 || c == 46 || (c >= 48 && c <= 57))//numeros y .
				num += (char)c;
			else if ((char)c == ',') {
				h_matriz[ifila*K + icolumna] = std::strtof(num.c_str(), NULL);
				num.clear();
				icolumna += 1;
			}
			else if ((char)c == ';') {
				h_matriz[ifila*K + icolumna] = std::strtof(num.c_str(), NULL);
				num.clear();
				ifila += 1;
				icolumna = 0;
			}
			else if ((char)c == '}') {
				h_matriz[ifila*K + icolumna] = std::strtof(num.c_str(), NULL);
				num.clear();
				icolumna += 1;
				ifila += 1;
			}
		}
		fclose(matrix_in);
	}


	//Operaciones para canal 0 (R);
	canal_in = 0;
	canal_out = 0;
	pasarHalo(H, L, K, R, numFilas, numColumnas, h_fotoPre, h_foto, image, canal_in);

	//Copiamos información de la foto desde h_foto

	cudaMemcpy(d_foto_in, h_foto, sizeof(float)*numColumnas*numFilas, cudaMemcpyHostToDevice);

	//Copiamos información de la matriz

	cudaMemcpy(d_matriz, h_matriz, sizeof(float)*K*K, cudaMemcpyHostToDevice);

	//kernel cuda
	kernel << <gridSize, blockSize, smBytes >> > (d_foto_out, d_foto_in, d_matriz, K, R, H, L, numFilas, numColumnas);

	//copiamos desde device

	cudaMemcpy(h_foto_out, d_foto_out, sizeof(float)*H*L, cudaMemcpyDeviceToHost);



	//imprimimos en formato png (en el canal que queremos
	for (i = 0; i<H; i++) {
		for (j = 0; j<L; j++) {
			if (h_foto_out[i*L + j] <= 0) h_foto_out[i*L + j] = 0;
			if (h_foto_out[i*L + j] >= 255) h_foto_out[i*L + j] = 255;
			image_out[4 * L * i + 4 * j + canal_out] = (unsigned char)h_foto_out[i*L + j];
		}
	}

	//Operaciones para canal 1 (G);
	canal_in = 1;
	canal_out = 1;
	pasarHalo(H, L, K, R, numFilas, numColumnas, h_fotoPre, h_foto, image, canal_in);

	//Copiamos información de la foto desde h_foto

	cudaMemcpy(d_foto_in, h_foto, sizeof(float)*numColumnas*numFilas, cudaMemcpyHostToDevice);

	//Copiamos información de la matriz

	cudaMemcpy(d_matriz, h_matriz, sizeof(float)*K*K, cudaMemcpyHostToDevice);

	//kernel cuda
	kernel << <gridSize, blockSize, smBytes >> > (d_foto_out, d_foto_in, d_matriz, K, R, H, L, numFilas, numColumnas);

	//copiamos desde device

	cudaMemcpy(h_foto_out, d_foto_out, sizeof(float)*H*L, cudaMemcpyDeviceToHost);

	//imprimimos en formato png (en el canal que queremos
	for (i = 0; i<H; i++) {
		for (j = 0; j<L; j++) {
			if (h_foto_out[i*L + j] <= 0) h_foto_out[i*L + j] = 0;
			if (h_foto_out[i*L + j] >= 255) h_foto_out[i*L + j] = 255;
			image_out[4 * L * i + 4 * j + canal_out] = (unsigned char)h_foto_out[i*L + j];
		}
	}

	//Operaciones para canal 2 (B);
	canal_in = 2;
	canal_out = 2;
	pasarHalo(H, L, K, R, numFilas, numColumnas, h_fotoPre, h_foto, image, canal_in);

	//Copiamos información de la foto desde h_foto

	cudaMemcpy(d_foto_in, h_foto, sizeof(float)*numColumnas*numFilas, cudaMemcpyHostToDevice);

	//Copiamos información de la matriz

	cudaMemcpy(d_matriz, h_matriz, sizeof(float)*K*K, cudaMemcpyHostToDevice);

	//kernel cuda
	kernel << <gridSize, blockSize, smBytes >> > (d_foto_out, d_foto_in, d_matriz, K, R, H, L, numFilas, numColumnas);

	//copiamos desde device

	cudaMemcpy(h_foto_out, d_foto_out, sizeof(float)*H*L, cudaMemcpyDeviceToHost);

	//imprimimos en formato png (en el canal que queremos
	for (i = 0; i<H; i++) {
		for (j = 0; j<L; j++) {
			if (h_foto_out[i*L + j] <= 0) h_foto_out[i*L + j] = 0;
			if (h_foto_out[i*L + j] >= 255) h_foto_out[i*L + j] = 255;
			image_out[4 * L * i + 4 * j + canal_out] = (unsigned char)h_foto_out[i*L + j];
		}
	}

	//Operaciones para canal 3 (A);
	canal_in = 3;
	canal_out = 3;
	pasarHalo(H, L, K, R, numFilas, numColumnas, h_fotoPre, h_foto, image, canal_in);

	//Copiamos información de la foto desde h_foto

	cudaMemcpy(d_foto_in, h_foto, sizeof(float)*numColumnas*numFilas, cudaMemcpyHostToDevice);

	//Copiamos información de la matriz

	cudaMemcpy(d_matriz, h_matriz, sizeof(float)*K*K, cudaMemcpyHostToDevice);

	//kernel cuda
	kernel << <gridSize, blockSize, smBytes >> > (d_foto_out, d_foto_in, d_matriz, K, R, H, L, numFilas, numColumnas);

	//copiamos desde device

	cudaMemcpy(h_foto_out, d_foto_out, sizeof(float)*H*L, cudaMemcpyDeviceToHost);

	//imprimimos en formato png (en el canal que queremos
	for (i = 0; i<H; i++) {
		for (j = 0; j<L; j++) {
			if (h_foto_out[i*L + j] <= 0) h_foto_out[i*L + j] = 0;
			if (h_foto_out[i*L + j] >= 255) h_foto_out[i*L + j] = 255;
			image_out[4 * L * i + 4 * j + canal_out] = image[4 * L * i + 4 * j + canal_out];//El canal A podemos cambiarlo tb, pero vamos a dejarlo como en la foto original
		}
	}


	//Encode the image
	error = lodepng::encode(filename_out, image_out, width, height);

	//if there's an error, display it
	if (error) std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;

	//imprimimos file con matriz
	FILE *fmatriz;
	fmatriz = fopen("matriz.txt", "w+");
	for (i = 0; i < K; i++) {
		for (j = 0; j < K; j++) {
			fprintf(fmatriz, "%f ", h_matriz[i*K + j]);
		}
		fprintf(fmatriz, "\n");
	}
	fprintf(fmatriz, "\n");
	fprintf(fmatriz, "\n");


	//Liberamos memoria 

	free(h_foto);
	free(h_fotoPre);
	free(h_matriz);
	free(h_foto_out);

	cudaFree(d_foto_in);
	cudaFree(d_matriz);
	cudaFree(d_foto_out);

	return(0);
}


/*FILE *fin;
FILE *fout;
FILE *fmatriz;

//NOTA: Modificar rutas de los ficheros si es necesario
fin = fopen("input.txt", "w+");
fout = fopen("output.txt", "w+");
fmatriz = fopen("matriz.txt", "w+");

fprintf(fin, "Cabecera fin \n");
fprintf(fout,"Cabecera fout \n");
fprintf(fmatriz, "Cabecera fmatriz \n");

for (i = 0; i < H; i++) {
for (j = 0; j < L; j++) {
fprintf(fin, "%f ", h_fotoPre[i*L + j]);
}
fprintf(fin, "\n");
}
fprintf(fin, "\n");
fprintf(fin, "\n");

for (i = 0; i < H; i++) {
for (j = 0; j < L; j++) {
fprintf(fout, "%f ", h_foto_out[i*L + j]);
}
fprintf(fout, "\n");
}
fprintf(fout, "\n");
fprintf(fout, "\n");

for (i = 0; i < K; i++) {
for (j = 0; j < K; j++) {
fprintf(fmatriz, "%f ", h_matriz[i*K + j]);
}
fprintf(fmatriz, "\n");
}
fprintf(fmatriz, "\n");
fprintf(fmatriz, "\n");
*/