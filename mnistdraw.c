// C_MNIST with interactive drawing and training integration

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <SDL2/SDL.h>
#include <termios.h>
#include <fcntl.h>

// Constants for network architecture and training
#define MAX_LAYERS 10                // Maximum number of layers in the neural network
#define MAX_NEURONS_PER_LAYER 512    // Maximum neurons in any single layer
#define INPUT_SIZE 784               // Size of input layer (28x28 pixels)
#define OUTPUT_SIZE 10               // Number of output classes (digits 0-9)

#define NUM_TRAINING_SAMPLES 60000   // Number of training samples in MNIST CSV
#define NUM_TEST_SAMPLES 10000       // Number of test samples in MNIST CSV
#define EPOCHS 10                    // Number of training epochs
#define BATCH_SIZE 1500              // Size of each mini-batch, adjusted to reduce batches per epoch
#define BATCH_PRINT_INTERVAL 5       // Interval at which batch results are printed

// Learning rate parameters
#define INITIAL_LEARNING_RATE 0.01   // Starting learning rate
#define DECAY_RATE 0.01              // Rate at which learning rate decays
#define MIN_LEARNING_RATE 0.0001     // Minimum learning rate

// User drawing weight factor
#define USER_DRAWING_WEIGHT 5.0      // Weight factor for user drawings

#define MAX_DISPLAY_IMAGES 10        // Maximum images to display

// Neural Network structure definition
typedef struct {
    int num_layers;                          // Total number of layers in the network
    int layer_sizes[MAX_LAYERS];             // Array to store the size of each layer
    double **weights;                        // 2D array for weights between layers
    double **biases;                         // 2D array for biases of each neuron
    double **activations;                    // 2D array for neuron activations
    double **z_values;                       // 2D array for weighted sums before activation
} NeuralNetwork;

// Function declarations
double adjust_learning_rate(double current_rate, int epoch);
void init_network(NeuralNetwork* nn, int num_layers, int* layer_sizes);
void free_network(NeuralNetwork* nn);
void forward_propagation(NeuralNetwork* nn, double* input);
void backward_propagation(NeuralNetwork* nn, double* target, double learning_rate, double weight_factor);
void* safe_malloc(size_t size);
double relu(double x);
double relu_derivative(double x);
void softmax(double* input, int length);
void load_csv_data(const char* filename, double*** images, int** labels, int num_samples, int image_size);
void shuffle_data(double** images, int* labels, int n);
void print_image(double* image, int width, int height);
void countdown(int seconds);
void flush_input();
int read_label();
void draw_digits_and_train(NeuralNetwork* nn, double*** train_images_ptr, int** train_labels_ptr, int* num_train_samples_ptr, double*** hand_drawn_images_ptr, int** hand_drawn_labels_ptr, int* num_hand_drawn_ptr);
void save_network(NeuralNetwork* nn, const char* filename);
void load_network(NeuralNetwork* nn, const char* filename);
void set_nonblocking_input(int enable);
int check_user_input();
void set_terminal_mode(int enable);
void handle_user_input(NeuralNetwork* nn, double*** train_images_ptr, int** train_labels_ptr, int* num_train_samples_ptr, double*** hand_drawn_images_ptr, int** hand_drawn_labels_ptr, int* num_hand_drawn_ptr, double** last_batch_images, int* last_batch_labels, int* num_last_batch_images);
void display_images(double** last_batch_images, int* last_batch_labels, int num_last_batch_images, double** hand_drawn_images, int* hand_drawn_labels, int num_hand_drawn);


int main() {
    // Seed the random number generator
    srand(time(NULL));

    // Load MNIST training data from CSV files
    double** train_images;
    int* train_labels;
    int num_train_samples = NUM_TRAINING_SAMPLES;
    load_csv_data("mnist_train.csv", &train_images, &train_labels, NUM_TRAINING_SAMPLES, INPUT_SIZE);

    // Load MNIST test data from CSV files
    double** test_images;
    int* test_labels;
    load_csv_data("mnist_test.csv", &test_images, &test_labels, NUM_TEST_SAMPLES, INPUT_SIZE);

    // Define the network architecture
    int num_layers = 4;
    int layer_sizes[] = {INPUT_SIZE, 128, 64, OUTPUT_SIZE};

    // Declare and initialize the neural network
    NeuralNetwork nn;
    init_network(&nn, num_layers, layer_sizes);

    // Set the initial learning rate
    double learning_rate = INITIAL_LEARNING_RATE;

    // Training parameters
    int epochs = EPOCHS;
    int batch_size = BATCH_SIZE;
    int num_batches = NUM_TRAINING_SAMPLES / batch_size;

    // Enable non-blocking input
    set_nonblocking_input(1);

    printf("\nYou can type 'draw' at any time to draw a digit, or 'display' to display images from the most recent batch, along with any hand-drawn images.\n\n");

    // Allocate memory to store recent batch data
    double** last_batch_images = (double**)malloc(MAX_DISPLAY_IMAGES * sizeof(double*));
    int* last_batch_labels = (int*)malloc(MAX_DISPLAY_IMAGES * sizeof(int));
    int num_last_batch_images = 0;

    // Allocate memory to store hand-drawn images
    double** hand_drawn_images = (double**)malloc(MAX_DISPLAY_IMAGES * sizeof(double*));
    int* hand_drawn_labels = (int*)malloc(MAX_DISPLAY_IMAGES * sizeof(int));
    int num_hand_drawn = 0;

    // Training loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        int correct = 0;

        // Shuffle the training data
        shuffle_data(train_images, train_labels, num_train_samples);

        for (int batch = 0; batch < num_batches; batch++) {

            // Handle user input (e.g., to draw a digit or display images)
            handle_user_input(&nn, &train_images, &train_labels, &num_train_samples, &hand_drawn_images, &hand_drawn_labels, &num_hand_drawn, last_batch_images, last_batch_labels, &num_last_batch_images);

            // Before processing the batch, free previous last_batch_images
            for (int i = 0; i < num_last_batch_images; i++) {
                free(last_batch_images[i]);
            }
            num_last_batch_images = 0;  // Reset the count of batch images

            // Initialize batch loss and correct count
            double batch_total_loss = 0.0;
            int batch_correct = 0;

            // Process each mini-batch
            for (int i = 0; i < batch_size; i++) {
                int index = batch * batch_size + i;
                double* input = train_images[index];
                int label = train_labels[index];

                // Perform forward propagation
                forward_propagation(&nn, input);

                // Create the target output vector (one-hot encoding)
                double target[OUTPUT_SIZE] = {0.0};
                target[label] = 1.0;

                // Calculate the loss for this example (cross-entropy loss)
                double loss = 0.0;
                if (nn.activations[num_layers - 1][label] > 0) {
                    loss = -log(nn.activations[num_layers - 1][label]);
                } else {
                    loss = -log(1e-15);  // Prevent log(0)
                }
                total_loss += loss;
                batch_total_loss += loss;

                // Perform backward propagation to update weights and biases
                backward_propagation(&nn, target, learning_rate, 1.0); // Weight factor 1.0 for normal training examples

                // Determine the network's prediction
                int predicted_label = 0;
                double max_activation = nn.activations[num_layers - 1][0];
                for (int j = 1; j < OUTPUT_SIZE; j++) {
                    if (nn.activations[num_layers - 1][j] > max_activation) {
                        max_activation = nn.activations[num_layers - 1][j];
                        predicted_label = j;
                    }
                }

                // Increment correct count if prediction matches label
                if (predicted_label == label) {
                    correct++;
                    batch_correct++;
                }

                // Store images and labels for display
                if (num_last_batch_images < MAX_DISPLAY_IMAGES) {
                    last_batch_images[num_last_batch_images] = (double*)malloc(INPUT_SIZE * sizeof(double));
                    memcpy(last_batch_images[num_last_batch_images], input, INPUT_SIZE * sizeof(double));
                    last_batch_labels[num_last_batch_images] = label;
                    num_last_batch_images++;
                }
            }

            // After processing the batch, check if we need to print batch results
            if (batch < 10 || (batch + 1) % BATCH_PRINT_INTERVAL == 0) {
                double batch_accuracy = (double)batch_correct / batch_size * 100;
                double batch_avg_loss = batch_total_loss / batch_size;

                printf("    - Batch %d, Accuracy: %.2f%%, Avg Loss: %.6f\n",
                       batch + 1, batch_accuracy, batch_avg_loss);
            }
        }

        // Calculate average loss for this epoch
        double avg_loss = total_loss / num_train_samples;

        // Adjust the learning rate based on the epoch
        learning_rate = adjust_learning_rate(learning_rate, epoch);

        // Print the formatted results for this epoch
        printf("\n\n========================================\n");
        printf("           Epoch %d Summary\n", epoch + 1);
        printf("========================================\n");
        printf("Accuracy       : %.2f%%\n", (double)correct / num_train_samples * 100);
        printf("Average Loss   : %.6f\n", avg_loss);
        printf("Learning Rate  : %.6f\n", learning_rate);
        printf("========================================\n");

        if (epoch < epochs - 1) {
            printf("Next epoch starting in: ");
            countdown(10);
            printf("\n\n");
        }
    }

    // Disable non-blocking input before proceeding
    set_nonblocking_input(0);

    // Free allocated memory for batch data
    for (int i = 0; i < num_last_batch_images; i++) {
        free(last_batch_images[i]);
    }
    free(last_batch_images);
    free(last_batch_labels);

    for (int i = 0; i < num_hand_drawn; i++) {
        free(hand_drawn_images[i]);
    }
    free(hand_drawn_images);
    free(hand_drawn_labels);

    // Testing phase
    printf("\nTesting the neural network:\n\n");
    int correct = 0;

    for (int i = 0; i < NUM_TEST_SAMPLES; i++) {
        double* input = test_images[i];
        int label = test_labels[i];

        // Perform forward propagation to get the network's prediction
        forward_propagation(&nn, input);

        // Determine the prediction
        int predicted_label = 0;
        double max_activation = nn.activations[num_layers - 1][0];
        for (int j = 1; j < OUTPUT_SIZE; j++) {
            if (nn.activations[num_layers - 1][j] > max_activation) {
                max_activation = nn.activations[num_layers - 1][j];
                predicted_label = j;
            }
        }

        // Increment correct count if prediction matches label
        if (predicted_label == label) correct++;
    }

    // Print the overall test accuracy
    printf("Test Accuracy: %.2f%%\n", (double)correct / NUM_TEST_SAMPLES * 100);

    // Allow the user to draw digits and classify them
    printf("\nYou can now draw digits and have the network classify them.\n");
    char response;
    do {
        draw_digits_and_train(&nn, NULL, NULL, NULL, NULL, NULL, NULL);
        printf("Would you like to draw another digit? (y/n): ");
        scanf(" %c", &response);
        flush_input(); // Clear any leftover input
    } while (response == 'y' || response == 'Y');

    // Free the memory allocated for the neural network and data
    free_network(&nn);

    for (int i = 0; i < num_train_samples; i++) {
        free(train_images[i]);
    }
    free(train_images);
    free(train_labels);

    for (int i = 0; i < NUM_TEST_SAMPLES; i++) {
        free(test_images[i]);
    }
    free(test_images);
    free(test_labels);

    return 0;
}

void countdown(int seconds) {
    for (int i = seconds; i > 0; i--) {
        printf("%d...", i);
        fflush(stdout);
        sleep(1);
    }
    printf("\n");
}

// Function to adjust the learning rate based on the current epoch
double adjust_learning_rate(double current_rate, int epoch) {
    // Calculate the new learning rate using the decay formula
    double new_rate = INITIAL_LEARNING_RATE / (1 + DECAY_RATE * epoch);

    // Ensure the new learning rate does not fall below the minimum learning rate
    if (new_rate < MIN_LEARNING_RATE) {
        new_rate = MIN_LEARNING_RATE;
    }

    // Return the adjusted learning rate
    return new_rate;
}

// Function to initialize the neural network
void init_network(NeuralNetwork* nn, int num_layers, int* layer_sizes) {
    // Check if the number of layers is valid
    if (num_layers < 2 || num_layers > MAX_LAYERS) {
        fprintf(stderr, "Invalid number of layers\n");
        exit(EXIT_FAILURE);
    }

    // Set the number of layers in the network structure
    nn->num_layers = num_layers;
    // Copy the layer sizes array into the network structure
    memcpy(nn->layer_sizes, layer_sizes, num_layers * sizeof(int));

    // Allocate memory for the weights, biases, activations, and z_values arrays
    nn->weights = (double**)safe_malloc((num_layers - 1) * sizeof(double*));
    nn->biases = (double**)safe_malloc((num_layers - 1) * sizeof(double*));
    nn->activations = (double**)safe_malloc(num_layers * sizeof(double*));
    nn->z_values = (double**)safe_malloc(num_layers * sizeof(double*));

    // Initialize weights and biases
    for (int i = 0; i < num_layers - 1; i++) {
        int current_layer_size = layer_sizes[i];
        int next_layer_size = layer_sizes[i + 1];

        // Allocate memory for weights and biases between current layer and next layer
        nn->weights[i] = (double*)safe_malloc(current_layer_size * next_layer_size * sizeof(double));
        nn->biases[i] = (double*)safe_malloc(next_layer_size * sizeof(double));

        // He initialization for weights
        double std_dev = sqrt(2.0 / current_layer_size);
        for (int j = 0; j < current_layer_size * next_layer_size; j++) {
            // Initialize weights with normal distribution (approximate using Box-Muller transform)
            double u1 = (double)rand() / RAND_MAX;
            double u2 = (double)rand() / RAND_MAX;
            double z = sqrt(-2.0 * log(u1 + 1e-9)) * cos(2.0 * M_PI * u2);
            nn->weights[i][j] = z * std_dev;
        }

        // Initialize biases to small positive values
        for (int j = 0; j < next_layer_size; j++) {
            nn->biases[i][j] = 0.01;
        }
    }

    // Allocate memory for activations and z_values
    for (int i = 0; i < num_layers; i++) {
        int layer_size = layer_sizes[i];
        nn->activations[i] = (double*)safe_malloc(layer_size * sizeof(double));
        nn->z_values[i] = (double*)safe_malloc(layer_size * sizeof(double));
    }
}

// Function to free all dynamically allocated memory for the neural network
void free_network(NeuralNetwork* nn) {
    // Free weights and biases
    for (int i = 0; i < nn->num_layers - 1; i++) {
        free(nn->weights[i]);
        free(nn->biases[i]);
    }

    // Free activations and z_values
    for (int i = 0; i < nn->num_layers; i++) {
        free(nn->activations[i]);
        free(nn->z_values[i]);
    }

    // Free arrays of pointers
    free(nn->weights);
    free(nn->biases);
    free(nn->activations);
    free(nn->z_values);
}

// Function to perform forward propagation through the neural network
void forward_propagation(NeuralNetwork* nn, double* input) {
    // Copy the input data to the activations of the input layer
    memcpy(nn->activations[0], input, nn->layer_sizes[0] * sizeof(double));

    // Loop through each layer starting from the first hidden layer
    for (int l = 1; l < nn->num_layers; l++) {
        int current_layer_size = nn->layer_sizes[l];
        int previous_layer_size = nn->layer_sizes[l - 1];

        // Compute activations for the current layer
        for (int j = 0; j < current_layer_size; j++) {
            double z = nn->biases[l - 1][j];  // Start with the bias

            // Sum the weighted activations from the previous layer
            for (int k = 0; k < previous_layer_size; k++) {
                double weight = nn->weights[l - 1][k * current_layer_size + j];
                double activation = nn->activations[l - 1][k];
                z += weight * activation;
            }

            nn->z_values[l][j] = z;  // Store the weighted sum before activation

            // Apply activation function
            if (l == nn->num_layers - 1) {
                // Output layer - will apply softmax after computing all z-values
                // Temporarily store z; softmax will be applied later
            } else {
                // Hidden layers - ReLU activation
                nn->activations[l][j] = relu(z);
            }
        }

        // If this is the output layer, apply softmax to the entire layer
        if (l == nn->num_layers - 1) {
            // Copy z_values to activations to apply softmax
            memcpy(nn->activations[l], nn->z_values[l], current_layer_size * sizeof(double));
            softmax(nn->activations[l], current_layer_size);
        }
    }
}

// Modified backward_propagation function to include weight_factor
void backward_propagation(NeuralNetwork* nn, double* target, double learning_rate, double weight_factor) {
    int num_layers = nn->num_layers;
    int* layer_sizes = nn->layer_sizes;
    double** activations = nn->activations;
    double** weights = nn->weights;
    double** biases = nn->biases;
    double** z_values = nn->z_values;

    // Allocate memory for delta values for each layer
    double** deltas = (double**)safe_malloc(num_layers * sizeof(double*));
    for (int l = 0; l < num_layers; l++) {
        deltas[l] = (double*)safe_malloc(layer_sizes[l] * sizeof(double));
    }

    // Step 1: Compute delta for the output layer
    int L = num_layers - 1;  // Index of the output layer
    for (int i = 0; i < layer_sizes[L]; i++) {
        double a = activations[L][i];  // Activation of the output neuron
        double y = target[i];          // Target value (one-hot encoded)
        // For softmax with cross-entropy loss, delta = a - y
        deltas[L][i] = (a - y) * weight_factor; // Multiply by weight factor
    }

    // Step 2: Backpropagate the error to hidden layers
    for (int l = L - 1; l > 0; l--) {
        int current_layer_size = layer_sizes[l];
        int next_layer_size = layer_sizes[l + 1];

        for (int i = 0; i < current_layer_size; i++) {
            double sum = 0.0;
            // Compute the weighted sum of deltas from the next layer
            for (int j = 0; j < next_layer_size; j++) {
                double weight = weights[l][i * next_layer_size + j];
                sum += weight * deltas[l + 1][j];
            }
            double z = z_values[l][i];  // Pre-activation value

            // Compute the delta for current layer neurons
            deltas[l][i] = sum * relu_derivative(z);
        }
    }

    // Step 3: Update weights and biases
    for (int l = 0; l < num_layers - 1; l++) {
        int current_layer_size = layer_sizes[l];
        int next_layer_size = layer_sizes[l + 1];

        // Update weights between layer l and l+1
        for (int i = 0; i < current_layer_size; i++) {
            for (int j = 0; j < next_layer_size; j++) {
                double delta = deltas[l + 1][j];
                double a = activations[l][i];
                // Gradient descent weight update
                weights[l][i * next_layer_size + j] -= learning_rate * a * delta;
            }
        }
        // Update biases for layer l+1
        for (int j = 0; j < next_layer_size; j++) {
            biases[l][j] -= learning_rate * deltas[l + 1][j];
        }
    }

    // Step 4: Free allocated memory for deltas
    for (int l = 0; l < num_layers; l++) {
        free(deltas[l]);
    }
    free(deltas);
}

// Safe memory allocation function
void* safe_malloc(size_t size) {
    // Allocate memory of the specified size
    void* ptr = malloc(size);

    // Check if memory allocation was successful
    if (ptr == NULL) {
        // If allocation failed, print an error message and exit the program
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Return the pointer to the allocated memory
    return ptr;
}

// ReLU activation function
double relu(double x) {
    return x > 0 ? x : 0;
}

// Derivative of the ReLU function
double relu_derivative(double x) {
    // x is the pre-activation value z
    return x > 0 ? 1.0 : 0.0;
}

// Softmax activation function
void softmax(double* input, int length) {
    double max = input[0];
    for (int i = 1; i < length; i++) {
        if (input[i] > max) {
            max = input[i];
        }
    }
    double sum = 0.0;
    for (int i = 0; i < length; i++) {
        input[i] = exp(input[i] - max);  // Subtract max for numerical stability
        sum += input[i];
    }
    for (int i = 0; i < length; i++) {
        input[i] /= sum;
    }
}

// Function to load MNIST data from a CSV file
void load_csv_data(const char* filename, double*** images, int** labels, int num_samples, int image_size) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        printf("Cannot open file: %s\n", filename);
        exit(1);
    }

    // Allocate memory for images and labels
    *images = (double**)malloc(num_samples * sizeof(double*));
    *labels = (int*)malloc(num_samples * sizeof(int));

    char line[16384];  // Adjust size according to the expected line length
    int sample = 0;
    while (fgets(line, sizeof(line), fp) && sample < num_samples) {
        char* token = strtok(line, ",");
        (*labels)[sample] = atoi(token);  // First value is the label

        (*images)[sample] = (double*)malloc(image_size * sizeof(double));
        int pixel = 0;
        while ((token = strtok(NULL, ",")) && pixel < image_size) {
            (*images)[sample][pixel] = atoi(token) / 255.0;  // Normalize pixel value
            pixel++;
        }
        sample++;
    }

    fclose(fp);
}

// Function to shuffle the training data
void shuffle_data(double** images, int* labels, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);

        // Swap images
        double* temp_image = images[i];
        images[i] = images[j];
        images[j] = temp_image;

        // Swap labels
        int temp_label = labels[i];
        labels[i] = labels[j];
        labels[j] = temp_label;
    }
}

// Function to print the image in ASCII format
void print_image(double* image, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            double pixel_value = image[i * width + j];
            if (pixel_value > 0.75) {
                printf("@");
            } else if (pixel_value > 0.5) {
                printf("#");
            } else if (pixel_value > 0.25) {
                printf("+");
            } else {
                printf(" ");
            }
        }
        printf("\n");
    }
}

void flush_input() {
    int c;
    while ((c = getchar()) != '\n' && c != EOF);
}

// Modify read_label() to read a single character without pressing Enter
int read_label() {
    int label = -1;
    char input_char;

    // Temporarily set terminal to non-canonical mode
    set_terminal_mode(1);

    printf("Please enter the correct label for your drawing (0-9): ");

    while (label < 0 || label > 9) {
        input_char = getchar();
        if (input_char >= '0' && input_char <= '9') {
            label = input_char - '0';
            printf("%c\n", input_char); // Echo the input
        } else {
            printf("\nInvalid input. Please enter a digit between 0 and 9: ");
        }
    }

    // Restore terminal to canonical mode
    set_terminal_mode(0);

    return label;
}

// Function to set terminal to non-canonical mode
void set_terminal_mode(int enable) {
    struct termios tty;
    tcgetattr(STDIN_FILENO, &tty);

    if (enable) {
        tty.c_lflag &= ~(ICANON | ECHO); // Turn off canonical mode and echo
        tty.c_cc[VMIN] = 1;
        tty.c_cc[VTIME] = 0;
    } else {
        tty.c_lflag |= ICANON | ECHO; // Turn on canonical mode and echo
    }
    tcsetattr(STDIN_FILENO, TCSANOW, &tty);
}

// Modify draw_digits_and_train() to use the updated backward_propagation()
void draw_digits_and_train(NeuralNetwork* nn, double*** train_images_ptr, int** train_labels_ptr, int* num_train_samples_ptr, double*** hand_drawn_images_ptr, int** hand_drawn_labels_ptr, int* num_hand_drawn_ptr) {
    const int window_width = 280;  // 10x the size of MNIST images
    const int window_height = 280;

    int continue_drawing = 1;

    while (continue_drawing) {
        // Initialize SDL
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            fprintf(stderr, "Could not initialize SDL2: %s\n", SDL_GetError());
            return;
        }

        SDL_Window* window = SDL_CreateWindow("Draw a digit (Press ESC or ENTER when done)", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, window_width, window_height, SDL_WINDOW_SHOWN);
        if (!window) {
            fprintf(stderr, "Could not create window: %s\n", SDL_GetError());
            SDL_Quit();
            return;
        }

        // Bring the window to the front
        SDL_RaiseWindow(window);

        SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
        if (!renderer) {
            fprintf(stderr, "Could not create renderer: %s\n", SDL_GetError());
            SDL_DestroyWindow(window);
            SDL_Quit();
            return;
        }

        // Set up the drawing surface
        SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_TARGET, window_width, window_height);
        if (!texture) {
            fprintf(stderr, "Could not create texture: %s\n", SDL_GetError());
            SDL_DestroyRenderer(renderer);
            SDL_DestroyWindow(window);
            SDL_Quit();
            return;
        }

        // Clear the drawing surface
        SDL_SetRenderTarget(renderer, texture);
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);  // Black background
        SDL_RenderClear(renderer);
        SDL_SetRenderTarget(renderer, NULL);

        int running = 1;
        int mouse_down = 0;

        printf("Draw your digit in the window. Press ESC or ENTER when done.\n");

        // Main event loop
        while (running) {
            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                switch (event.type) {
                    case SDL_QUIT:
                        running = 0;
                        continue_drawing = 0;
                        break;
                    case SDL_MOUSEBUTTONDOWN:
                        if (event.button.button == SDL_BUTTON_LEFT) {
                            mouse_down = 1;
                        }
                        break;
                    case SDL_MOUSEBUTTONUP:
                        if (event.button.button == SDL_BUTTON_LEFT) {
                            mouse_down = 0;
                        }
                        break;
                    case SDL_MOUSEMOTION:
                        if (mouse_down) {
                            // Draw a circle at mouse position
                            int x = event.motion.x;
                            int y = event.motion.y;
                            int brush_size = 12;  // Adjust as needed

                            SDL_SetRenderTarget(renderer, texture);
                            SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);  // White color
                            for (int w = 0; w < brush_size; w++) {
                                for (int h = 0; h < brush_size; h++) {
                                    int dx = x + w - brush_size / 2;
                                    int dy = y + h - brush_size / 2;
                                    if (dx >= 0 && dx < window_width && dy >= 0 && dy < window_height) {
                                        SDL_RenderDrawPoint(renderer, dx, dy);
                                    }
                                }
                            }
                            SDL_SetRenderTarget(renderer, NULL);
                        }
                        break;
                    case SDL_KEYDOWN:
                        if (event.key.keysym.sym == SDLK_RETURN || event.key.keysym.sym == SDLK_KP_ENTER || event.key.keysym.sym == SDLK_ESCAPE) {
                            running = 0;
                            if (event.key.keysym.sym == SDLK_ESCAPE) {
                                continue_drawing = 0;
                            }
                        }
                        break;
                }
            }

            // Render the texture to the screen
            SDL_RenderClear(renderer);
            SDL_RenderCopy(renderer, texture, NULL, NULL);
            SDL_RenderPresent(renderer);
        }

        if (!continue_drawing) {
            // Free resources
            SDL_DestroyTexture(texture);
            SDL_DestroyRenderer(renderer);
            SDL_DestroyWindow(window);
            SDL_Quit();
            break;
        }

        // Get the pixel data from the texture
        Uint32* pixels = (Uint32*)malloc(window_width * window_height * sizeof(Uint32));
        if (!pixels) {
            fprintf(stderr, "Could not allocate memory for pixels\n");
            SDL_DestroyTexture(texture);
            SDL_DestroyRenderer(renderer);
            SDL_DestroyWindow(window);
            SDL_Quit();
            return;
        }

        SDL_SetRenderTarget(renderer, texture);
        SDL_RenderReadPixels(renderer, NULL, SDL_PIXELFORMAT_ARGB8888, pixels, window_width * sizeof(Uint32));
        SDL_SetRenderTarget(renderer, NULL);

        // Now, process the pixels to create a 28x28 grayscale image
        // We need to resize the window_width x window_height image to 28x28

        double input_image[INPUT_SIZE];  // INPUT_SIZE = 784

        // Simple downsampling: average blocks of pixels
        int block_width = window_width / 28;
        int block_height = window_height / 28;

        SDL_PixelFormat* format = SDL_AllocFormat(SDL_PIXELFORMAT_ARGB8888);

        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                double sum = 0.0;
                for (int y = i * block_height; y < (i + 1) * block_height; y++) {
                    for (int x = j * block_width; x < (j + 1) * block_width; x++) {
                        Uint32 pixel = pixels[y * window_width + x];
                        Uint8 r, g, b, a;
                        SDL_GetRGBA(pixel, format, &r, &g, &b, &a);
                        // Since the drawing is in white on black, we can use the red channel
                        // Normalize to 0-1
                        sum += r / 255.0;
                    }
                }
                // Average pixel value
                sum /= (block_width * block_height);
                // Invert the pixel value because drawing is white on black background
                input_image[i * 28 + j] = sum;
            }
        }

        SDL_FreeFormat(format);

        // Now, feed input_image to the neural network
        forward_propagation(nn, input_image);

        // Determine the network's prediction
        int predicted_label = 0;
        double max_activation = nn->activations[nn->num_layers - 1][0];
        for (int j = 1; j < OUTPUT_SIZE; j++) {
            if (nn->activations[nn->num_layers - 1][j] > max_activation) {
                max_activation = nn->activations[nn->num_layers - 1][j];
                predicted_label = j;
            }
        }

        printf("\nThe neural network predicts your drawing as: %d\n", predicted_label);

        // Ask the user to confirm or correct the label
        int label = read_label();

        // If training data pointers are provided, add the drawn image to training data
        if (train_images_ptr && train_labels_ptr && num_train_samples_ptr) {
            int num_train_samples = *num_train_samples_ptr;
            double** train_images = *train_images_ptr;
            int* train_labels = *train_labels_ptr;

            // Reallocate memory to accommodate the new sample
            train_images = realloc(train_images, (num_train_samples + 1) * sizeof(double*));
            train_labels = realloc(train_labels, (num_train_samples + 1) * sizeof(int));
            if (!train_images || !train_labels) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(EXIT_FAILURE);
            }

            // Add the new sample
            train_images[num_train_samples] = malloc(INPUT_SIZE * sizeof(double));
            memcpy(train_images[num_train_samples], input_image, INPUT_SIZE * sizeof(double));
            train_labels[num_train_samples] = label;
            num_train_samples++;

            // Update the pointers
            *train_images_ptr = train_images;
            *train_labels_ptr = train_labels;
            *num_train_samples_ptr = num_train_samples;

            // Also store in hand_drawn_images and hand_drawn_labels
            if (hand_drawn_images_ptr && hand_drawn_labels_ptr && num_hand_drawn_ptr) {
                double** hand_drawn_images = *hand_drawn_images_ptr;
                int* hand_drawn_labels = *hand_drawn_labels_ptr;
                int num_hand_drawn = *num_hand_drawn_ptr;

                if (num_hand_drawn < MAX_DISPLAY_IMAGES) {
                    hand_drawn_images[num_hand_drawn] = (double*)malloc(INPUT_SIZE * sizeof(double));
                    memcpy(hand_drawn_images[num_hand_drawn], input_image, INPUT_SIZE * sizeof(double));
                    hand_drawn_labels[num_hand_drawn] = label;
                    num_hand_drawn++;
                } else {
                    // Shift images to make room for the new one
                    free(hand_drawn_images[0]);
                    for (int i = 1; i < MAX_DISPLAY_IMAGES; i++) {
                        hand_drawn_images[i - 1] = hand_drawn_images[i];
                        hand_drawn_labels[i - 1] = hand_drawn_labels[i];
                    }
                    hand_drawn_images[MAX_DISPLAY_IMAGES - 1] = (double*)malloc(INPUT_SIZE * sizeof(double));
                    memcpy(hand_drawn_images[MAX_DISPLAY_IMAGES - 1], input_image, INPUT_SIZE * sizeof(double));
                    hand_drawn_labels[MAX_DISPLAY_IMAGES - 1] = label;
                }

                *hand_drawn_images_ptr = hand_drawn_images;
                *hand_drawn_labels_ptr = hand_drawn_labels;
                *num_hand_drawn_ptr = num_hand_drawn;
            }

            // Train on the new sample immediately with higher weight
            double target[OUTPUT_SIZE] = {0.0};
            target[label] = 1.0;
            backward_propagation(nn, target, INITIAL_LEARNING_RATE, USER_DRAWING_WEIGHT);

            printf("The network has been trained on your drawing with increased weight.\n");
        } else {
            // If not in training mode, just classify
            printf("Classification complete.\n");
        }

        free(pixels);

        // Free resources for this iteration
        SDL_DestroyTexture(texture);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();

        // Relaunch the window if the user wants to continue drawing
        printf("Would you like to draw another digit? (y/n): ");
        char response;
        scanf(" %c", &response);
        flush_input(); // Clear any leftover input
        if (response != 'y' && response != 'Y') {
            continue_drawing = 0;
        }
    }
}

// Function to save the network state to a file
void save_network(NeuralNetwork* nn, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Could not open file to save network state\n");
        return;
    }

    // Save the number of layers and layer sizes
    fwrite(&nn->num_layers, sizeof(int), 1, file);
    fwrite(nn->layer_sizes, sizeof(int), nn->num_layers, file);

    // Save weights and biases
    for (int i = 0; i < nn->num_layers - 1; i++) {
        int current_layer_size = nn->layer_sizes[i];
        int next_layer_size = nn->layer_sizes[i + 1];
        fwrite(nn->weights[i], sizeof(double), current_layer_size * next_layer_size, file);
        fwrite(nn->biases[i], sizeof(double), next_layer_size, file);
    }

    fclose(file);
}

// Function to load the network state from a file
void load_network(NeuralNetwork* nn, const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Could not open file to load network state\n");
        return;
    }

    int num_layers;
    fread(&num_layers, sizeof(int), 1, file);

    if (num_layers != nn->num_layers) {
        fprintf(stderr, "Network structure mismatch\n");
        fclose(file);
        return;
    }

    int* layer_sizes = (int*)malloc(num_layers * sizeof(int));
    fread(layer_sizes, sizeof(int), num_layers, file);

    for (int i = 0; i < num_layers; i++) {
        if (layer_sizes[i] != nn->layer_sizes[i]) {
            fprintf(stderr, "Layer size mismatch\n");
            free(layer_sizes);
            fclose(file);
            return;
        }
    }
    free(layer_sizes);

    // Load weights and biases
    for (int i = 0; i < nn->num_layers - 1; i++) {
        int current_layer_size = nn->layer_sizes[i];
        int next_layer_size = nn->layer_sizes[i + 1];
        fread(nn->weights[i], sizeof(double), current_layer_size * next_layer_size, file);
        fread(nn->biases[i], sizeof(double), next_layer_size, file);
    }

    fclose(file);
}

// Functions to enable and disable non-blocking input
void set_nonblocking_input(int enable) {
#ifdef _WIN32
    // Windows-specific code
    // Not implemented here
#else
    struct termios ttystate;
    tcgetattr(STDIN_FILENO, &ttystate);

    if (enable) {
        ttystate.c_lflag &= ~ICANON; // turn off canonical mode
        ttystate.c_lflag &= ~ECHO;   // turn off echo
        ttystate.c_cc[VMIN] = 0;
        ttystate.c_cc[VTIME] = 0;
    } else {
        ttystate.c_lflag |= ICANON; // turn on canonical mode
        ttystate.c_lflag |= ECHO;   // turn on echo
    }
    tcsetattr(STDIN_FILENO, TCSANOW, &ttystate);
#endif
}

int check_user_input() {
    char buffer[16];
    int n = read(STDIN_FILENO, buffer, sizeof(buffer) - 1);
    if (n > 0) {
        buffer[n] = '\0';
        for (int i = 0; i < n; i++) {
            buffer[i] = tolower(buffer[i]);  // Convert to lowercase
        }
        if (strstr(buffer, "draw") != NULL) {
            return 1; // User typed 'draw'
        } else if (strstr(buffer, "display") != NULL) {
            return 2; // User typed 'display'
        }
    }
    return 0;
}

void handle_user_input(NeuralNetwork* nn, double*** train_images_ptr, int** train_labels_ptr, int* num_train_samples_ptr, double*** hand_drawn_images_ptr, int** hand_drawn_labels_ptr, int* num_hand_drawn_ptr, double** last_batch_images, int* last_batch_labels, int* num_last_batch_images) {
    int user_input = check_user_input();
    if (user_input == 1) {  // User typed "draw"
        printf("\nEntering drawing mode...\n");
        
        // Save the network state
        save_network(nn, "network_state.dat");

        // Disable non-blocking input before entering SDL
        set_nonblocking_input(0);

        // Enter drawing mode
        draw_digits_and_train(nn, train_images_ptr, train_labels_ptr, num_train_samples_ptr, hand_drawn_images_ptr, hand_drawn_labels_ptr, num_hand_drawn_ptr);

        // Reload the network state
        load_network(nn, "network_state.dat");

        // Re-enable non-blocking input
        set_nonblocking_input(1);

        printf("\nResuming training...\n\n");
    } else if (user_input == 2) { // User typed "display"
        // Call the function to display images
        display_images(last_batch_images, last_batch_labels, *num_last_batch_images, *hand_drawn_images_ptr, *hand_drawn_labels_ptr, *num_hand_drawn_ptr);
    }
}

// Function to display training images, including hand-drawn ones if any
// Function to display training images, including hand-drawn ones if any
void display_images(double** last_batch_images, int* last_batch_labels, int num_last_batch_images, double** hand_drawn_images, int* hand_drawn_labels, int num_hand_drawn) {
    printf("\nDisplaying images:\n");

    // Display batch images
    int num_batch_display = num_last_batch_images < MAX_DISPLAY_IMAGES ? num_last_batch_images : MAX_DISPLAY_IMAGES;
    for (int i = 0; i < num_batch_display; i++) {
        printf("\nBatch Image %d (Label: %d):\n", i + 1, last_batch_labels[i]);
        print_image(last_batch_images[i], 28, 28);
    }

    // Display hand-drawn images
    int num_hand_drawn_display = num_hand_drawn < MAX_DISPLAY_IMAGES ? num_hand_drawn : MAX_DISPLAY_IMAGES;
    for (int i = 0; i < num_hand_drawn_display; i++) {
        printf("\nHand-Drawn Image %d (Label: %d):\n", i + 1, hand_drawn_labels[i]);
        print_image(hand_drawn_images[i], 28, 28);
    }
}
