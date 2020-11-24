/**
*
* Pomocne funkce.
*
*/

#pragma once

#include    <vector>
#include    <string>
#include    <iostream>
#include    "../constants.h"

namespace kiv_ppr_utils {

    // --- Hodnota glykemie mmol/L, pod kterou je nutna lekarska pozornost ---
    static constexpr double Low_Threshold = 3.0;

    // --- Hodnota glykemie mmol/L, nad kterou je nutna lekarska pozornost ---
    static constexpr double High_Threshold = 13.0;

    // --- Pocet neuronu uvnitr krajnich hranic ---
    static constexpr size_t Internal_Bound_Count = OUTPUT_LAYER_NEURONS_COUNT - 2;

    // --- Velikost intervalu ---
    static constexpr double Band_Size =
        (High_Threshold - Low_Threshold) / static_cast<double>(Internal_Bound_Count);

    // --- Invertovana hodnota intervalu ---
    static constexpr double Inv_Band_Size = 1.0 / Band_Size;

    // --- Polovina intervalu ---
    static constexpr double Half_Band_Size = 0.5 / Inv_Band_Size;

    // --- Pocet vsech intervalu (odpovida poctu vystupnich neuronu) ---
    static constexpr size_t Band_Count = Internal_Bound_Count + 2;

	
    /**
    * Priradi hodnotu indexu vystupniho neuronu do spravneho intervalu.
    *
    * params:
    *   index - index vystupniho neuronu
    *
    * return:
    *   stredni hodnota intervalu
    */
    double Band_Index_To_Level(const size_t index);

    /**
    * Vypocita index neuronu ve vystupni vrstve pro ocekavanou hodnotu.
    *
    * params:
    *   expected_value - ocekavana hodnota
    *
    * return:
    *   index neuronu vystupni vrstvy
    */
	size_t Band_Level_To_Index(double expected_value);

    /**
    * Naskaluje vstupni hodnotu do intervalu <-1, 1>
    *
    * params:
    *   bg - vstupni hodnota glykemie
    *
    * return:
    *   hodnota z intervalu <-1, 1>
    */
    double Risk_Function(const double bg);

    /**
    * Vytvori vektor hodnot 0 a 1, kde 1 je na miste ocekavane hodnoty.
    *
    * params:
    *   expected_value - ocekavana hodnota
    *
    * return:
    *   vektor hodnot 0 a 1
    */
    std::vector<double> Get_Target_Values_Vector(double expected_value);

    /**
    * Generator pseudonahodnych cisel dle uniformniho rozdeleni (generuje hodnoty od 0 do 1)
    *
    * return:
    *   vygenerovana hodnota
    */
    double Get_Random_Weight();

    /**
    * Vypocita prumernou relativni chybu.
    *
    * params:
    *   relative_errors_vector - vektor relativnich chyb
    *
    * return:
    *   prumerna relativni chyba
    */
    double Calc_Average_Relative_Error(std::vector<double> relative_errors_vector);

    /**
    * Vypocita prumernou odchylku relativnich chyb.
    *
    * params:
    *   relative_errors_vector - vektor relativnich chyb
    *
    * return:
    *   prumerna odchylka relativnich chyb
    */
    double Calc_Standard_Deviation(std::vector<double> relative_errors_vector);

    /**
    * Vypocita celkovou chybu relativnich chyb (prumerna chyba + odchylka)
    *
    * params:
    *   relative_errors_vector - vektor relativnich chyb
    *
    * return:
    *   celkova chyba
    */
    double Calculate_Total_Error(std::vector<double> relative_errors_vector);

    /**
    * Vygeneruje retezec relativnich chyb (pro generovani CSV souboru).
    *
    * params:
    *   relative_errors_vector - vektor relativnich chyb
    *
    * return:
    *   retezec relativnich chyb
    */
    std::string Generate_Csv(std::vector<double> relative_errors_vector);

    /**
    * Vygeneruje retezec vysledku predikce (pro generovani TXT souboru vysledku).
    *
    * params:
    *   input_values - vektor vstupnich hodnot
    *   result_values - vektor vysledku
    *   expected_values - vektor ocekavanych hodnot
    *   input_layer_neurons_count - pocet neuronu vstupni vrstcy
    *
    * return:
    *   retezec vysledku predikce
    */
    std::string Generate_Result_File(std::vector<double>& input_values, 
        std::vector<double>& result_values, 
        std::vector<double>& expected_values, 
        unsigned input_layer_neurons_count);
}