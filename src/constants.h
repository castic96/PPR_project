/**
*
* Hlavni konstanty programu.
*
*/

#pragma once

// --- Inicializacni hodnota id pro parsovani vstupnich dat ---
#define		START_ID		                -1

// --- Pocet neuronu vstupni vrstvy ---
#define     INPUT_LAYER_NEURONS_COUNT       8

// --- Pocet neuronu prvni skryte vrstvy ---
#define     HIDDEN1_LAYER_NEURONS_COUNT     16

// --- Pocet neuronu druhe skryte vrstvy ---
#define     HIDDEN2_LAYER_NEURONS_COUNT     26

// --- Pocet neuronu vystupni vrstvy ---
#define     OUTPUT_LAYER_NEURONS_COUNT      32

// --- Pocet siti k trenovani na CPU ---
#define     NEURAL_NETWORKS_COUNT           20

// --- Casovy rozestup mezi namerenymi hodnotami v minutach ---
#define     MEASURE_INTERVAL_MINUTES		5

// --- Umisteni souboru s kodem pro OpenCL ---
#define		CL_FILE_DEST					"..\\src\\gpu\\network_gpu.cl"

// --- Cesta k ulozeni CSV souboru s relativnimi chybami ---
#define     CSV_PATH						"../out/errors.csv"

// --- Cesta k ulozeni INI souboru s parametry neuronove site ---
#define     NEURAL_INI_PATH					"../out/neural.ini"

// --- Cesta k ulozeni TXT souboru s vysledky predikce ---
#define     RESULTS_PATH					"../out/results.txt"

// --- Cesta k ulozeni SVG souboru se zelenym grafem neuronove site ---
#define     GREEN_GRAPH_PATH				"../out/green_graph.svg"

// --- Cesta k ulozeni SVG souboru s modrym grafem neuronove site ---
#define     BLUE_GRAPH_PATH					"../out/blue_graph.svg"