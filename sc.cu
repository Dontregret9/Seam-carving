#include <stdio.h>
#include <stdint.h>

#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
		fprintf(stderr, "code: %d, reason: %s\n", error,\
				cudaGetErrorString(error));\
		exit(EXIT_FAILURE);\
	}\
}

struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start, 0);                                                                 
		cudaEventSynchronize(start);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
	}

	float Elapsed()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};

void readPnm(char * fileName, 
		int &numChannels, int &width, int &height, uint8_t * &pixels)
{
	FILE * f = fopen(fileName, "r");
	if (f == NULL)
	{
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	char type[3];
	fscanf(f, "%s", type);
	if (strcmp(type, "P2") == 0)
		numChannels = 1;
	else if (strcmp(type, "P3") == 0)
		numChannels = 3;
	else // In this exercise, we don't touch other types
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	fscanf(f, "%i", &width);
	fscanf(f, "%i", &height);

	int max_val;
	fscanf(f, "%i", &max_val);
	if (max_val > 255) // In this exercise, we assume 1 byte per value
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	pixels = (uint8_t *)malloc(width * height * numChannels);
	for (int i = 0; i < width * height * numChannels; i++)
		fscanf(f, "%hhu", &pixels[i]);

	fclose(f);
}

void writePnm(uint8_t * pixels, int numChannels, int width, int height, 
		char * fileName)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

	if (numChannels == 1)
		fprintf(f, "P2\n");
	else if (numChannels == 3)
		fprintf(f, "P3\n");
	else
	{
		fclose(f);
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	fprintf(f, "%i\n%i\n255\n", width, height); 

	for (int i = 0; i < width * height * numChannels; i++)
		fprintf(f, "%hhu\n", pixels[i]);

	fclose(f);
}


void convertRgb2Gray(uint8_t * inPixels, int width, int height,
		uint8_t * outPixels, 	
		bool useDevice=false, dim3 blockSize=dim3(1))
{
	GpuTimer timer;
	timer.Start();		
	if (useDevice == false)
	{
        // Reminder: gray = 0.299*red + 0.587*green + 0.114*blue  
        for (int r = 0; r < height; r++)
        {
            for (int c = 0; c < width; c++)
            {
                int i = r * width + c;
                uint8_t red = inPixels[3 * i];
                uint8_t green = inPixels[3 * i + 1];
				uint8_t blue = inPixels[3 * i + 2];
				outPixels[i] = 0.299f*red + 0.587f*green + 0.114f*blue;
            }
        }
	}

	timer.Stop();
	float time = timer.Elapsed();
	printf("Processing time (%s): %f ms\n\n", 
			useDevice == true? "use device" : "use host", time);
}


char * concatStr(const char * s1, const char * s2)
{
	char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
	strcpy(result, s1);
	strcat(result, s2);
	return result;
}


void writeEnergyMatrix(float * matrix, int width, int height, char * fileName)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}
	for (int r = 0; r < height; r++)
	{
		for (int c = 0; c < width; c++)
		{
			int i = r*width + c;
			fprintf(f, "%f ", matrix[i]);
		}
		fprintf(f, "\n");
	}
	fclose(f);

}

void convolution(uint8_t * inPixels, int width, int height, float * filter, int filterWidth, float * outPixels)
{
    for (int outPixelsR = 0; outPixelsR < height; outPixelsR++)
		{
			for (int outPixelsC = 0; outPixelsC < width; outPixelsC++)
			{
				float outPixel = 0;
				for (int filterR = 0; filterR < filterWidth; filterR++)
				{
					for (int filterC = 0; filterC < filterWidth; filterC++)
					{
						float filterVal = filter[filterR*filterWidth + filterC];
						int inPixelsR = outPixelsR - filterWidth/2 + filterR;
						int inPixelsC = outPixelsC - filterWidth/2 + filterC;
						inPixelsR = min(max(0, inPixelsR), height - 1);
						inPixelsC = min(max(0, inPixelsC), width - 1);
						int inPixel = inPixels[inPixelsR*width + inPixelsC];
						outPixel += (float)filterVal * (float)inPixel;
					}
				}
				outPixels[outPixelsR*width + outPixelsC] = outPixel;
			}
		}
}

void computeEnergy(uint8_t * inPixels, int width, int height,float* gradX, float* gradY, float * outPixels)
{
	int filterWidth = 3;
	float sobelX[9] = {1,0,-1,2,0,-2,1,0,-1};
	float sobelY[9] = {1,2,1,0,0,0,-1,-2,-1};

	
	convolution(inPixels, width, height, sobelX, filterWidth, gradX);

	
	convolution(inPixels, width, height, sobelY, filterWidth, gradY);

	for (int r = 0; r < height; r++)
	{
		for (int c = 0; c < width; c++)
		{
			int i = r*width + c;
			outPixels[i] = abs(gradX[i]) + abs(gradY[i]);
		}
	}
}

void createCumulativeEnergyMap(float* energy, int width, int height, float* table)
{
	int _size = height * width;
	float* lastRowEnergy = &energy[_size -1];
	float* lastRowTable = &table[_size - 1];
	// Copy dong cuoi cung cua energy
	for (int p = width ; p > 0; p--)
	{
		*(table + _size - p ) = *(lastRowEnergy - p + 1);
	}
	// Duyet mang energy
	for (int row = height - 2; row  >=  0; row--, lastRowEnergy -= width , lastRowTable -= width)
	{
		float* pRow = lastRowEnergy;
		float* pRowTable = lastRowTable;
		// Xet vi tri dau cua mot dong
		pRowTable[-width ] = pRow[-width] + min(pRowTable[0] , pRowTable[-1] );
		// Xet cac vi khac
		for (int col = 2; col < width ; col++)
		{
			*(pRowTable -col - width + 1) = *(pRow - col - width + 1) + min(min(pRowTable[-col + 1] , pRowTable[-col] ), pRowTable[-col + 2] );
		}
		// Xet vi tri cuoi
		pRowTable[-width - width + 1] = pRow[-width - width + 1] + min(pRowTable[-width + 1] , pRowTable[-width + 2] );
	}
}


void findOptSeam(float * table, int width, int height, float * optSeam)
{
	int tmp;
	float* pTable = table;
	// Tim phan tu nho nhat trong dong thu 0
	int minVal = pTable[0];
	int minPos = 0;
	
	for (int i = 0; i < width; i++)
	{
		if (pTable[i] < minVal)
		{
			minVal = pTable[i];
			minPos = i;
		}
	}
	optSeam[0] = minPos;
	pTable += width;
	// Duyet qua các dong
	for (int row = 1; row < height; row++, pTable +=width)
	{
		if (minPos == 0) //  o dau
		{
			minPos = pTable[0] < pTable[1] ? 0 : 1;
			
		}
		else if(minPos == width - 1) // o cuoi
		{
			minPos = pTable[width - 1] < pTable[width - 2] ? width - 1 : width - 2;
		}
		else // o giua
		{
			tmp = pTable[minPos] < pTable[minPos + 1] ? minPos : minPos + 1;
			if (pTable[tmp] > pTable[minPos - 1])
				minPos = minPos - 1;
			else
				minPos = tmp;
		}
		optSeam[row] = minPos;
	}
}

void deleteOptSeam(uint8_t * inPixels, float* table, float* eneryMatrix, float* chosenSeam, int height, int cur_width)
{
	// // convert position in row --> position in all matrix
	// for(int i=0;i<height;i++0)
	// {
	// 	chosenSeam[i] += i*cur_width;

	// 	for(int j=chosenSeam[i]-i; j<cur_width*height-i-1;j++)
	// 	{
	// 		table[j] = table[j+1];
	// 		energyMatrix[j] = energyMatrix[j+1]

	// 		//delete in P3 color
			
	// 	}
	// }
}

int main(int argc, char ** argv)
{	
	if (argc != 4 && argc != 6)
	{
		printf("The number of arguments is invalid\n");
		return EXIT_FAILURE;
	}
	
	// Read input RGB image file
	int numChannels, width, height;
	uint8_t * inPixels;
	readPnm(argv[1], numChannels, width, height, inPixels);
	if (numChannels != 3)
		return EXIT_FAILURE; // Input image must be RGB
	printf("Image size (width x height): %i x %i\n\n", width, height);

	// Convert RGB to grayscale not using device
	uint8_t * correctOutPixels= (uint8_t *)malloc(width * height);
	convertRgb2Gray(inPixels, width, height, correctOutPixels);

	// Write results to files
	char * outFileNameBase = strtok(argv[2], "."); // Get rid of extension
	writePnm(correctOutPixels, 1, width, height, concatStr(outFileNameBase, "_host.pnm"));

    // Here correctOutPixels is grayscale image

    float* gradX = (float*)malloc(width*height*sizeof(float));
    float* gradY = (float*)malloc(width*height*sizeof(float));
    float* energyMatrix = (float*)malloc(width*height*sizeof(float));
    computeEnergy(correctOutPixels, width, height, gradX, gradY, energyMatrix);
    
    writeEnergyMatrix(energyMatrix, width, height, argv[3]);
	writeEnergyMatrix(gradX, width, height, argv[4]);
    writeEnergyMatrix(gradY, width, height, argv[5]);
    
	float* table = (float*)malloc(width*height*sizeof(float));
	float* optSeam = (float*)malloc(height*sizeof(float));

    // int width_expect = argv[6];
    // int cur_width = width;
    // while(width_expect < cur_width)
    // {
    //     // find the seam will be remove
    //     createCumulativeEnergyMap(energyMatrix, width, height, table);
    //     writeEnergyMatrix(table, width, height, argv[5]);

    //     findOptSeam(table, width, height, optSeam);
    //     // remove this seam (in eneryMatrix & table)
	// 	deleteOptSeam(inPixels, table, energyMatrix, cur_width, height, optSeam);
	// 	cur_width--;
    // }




    
    // Free memories
    free(gradX);
	free(gradY);
    free(inPixels);
    
	free(energyMatrix);
	free(table);
	free(optSeam);
}
