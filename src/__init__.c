#include "py/obj.h"
#include "py/runtime.h"
#include "py/mpconfig.h"
#include "py/binary.h"


// ML model entry point (from model.c)
void entry(const float tensor_input[1][30][30][1],
           float tensor_output[1][10]);



/* -------------------------------------------------- */
/* NEW: invoke(input_array)                     */
/* -------------------------------------------------- */

static mp_obj_t digit_classifier_invoke(mp_obj_t input_obj) {

    mp_buffer_info_t input_buf;
    mp_get_buffer_raise(input_obj, &input_buf, MP_BUFFER_READ);

    // Expect 900 floats (30*30)
    if (input_buf.len != 900 * sizeof(float)) {
        mp_raise_ValueError(
            MP_ERROR_TEXT("Input must be 900 floats")
        );
    }

    const float *input = (const float *)input_buf.buf;

    // --- CHANGE STARTS HERE ---
    
    // 1. Create a Python bytearray large enough for 10 floats
    size_t output_size = 10 * sizeof(float);
    mp_obj_t output_obj = mp_obj_new_bytearray(output_size, NULL);

    // 2. Get the pointer to the underlying buffer of the new bytearray
    mp_buffer_info_t output_buf;
    mp_get_buffer_raise(output_obj, &output_buf, MP_BUFFER_WRITE);
    float *output = (float *)output_buf.buf;

    // --- CHANGE ENDS HERE ---

    // Call ML model
    entry(
        (const float (*)[30][30][1])input,
        (float (*)[10])output
    );

    return output_obj;
}

static MP_DEFINE_CONST_FUN_OBJ_1(digit_classifier_invoke_obj,
                                digit_classifier_invoke);

/* -------------------------------------------------- */
/* Module globals                                     */
/* -------------------------------------------------- */

static const mp_rom_map_elem_t digit_classifier_module_globals_table[] = {
    { MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_digit_classifier) },
    { MP_ROM_QSTR(MP_QSTR_invoke), MP_ROM_PTR(&digit_classifier_invoke_obj) },
    { MP_ROM_QSTR(MP_QSTR_invoke),
      MP_ROM_PTR(&digit_classifier_invoke_obj) },
};

static MP_DEFINE_CONST_DICT(digit_classifier_module_globals,
                            digit_classifier_module_globals_table);

/* -------------------------------------------------- */
/* Module definition                                  */
/* -------------------------------------------------- */

const mp_obj_module_t digit_classifier_module = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t *)&digit_classifier_module_globals,
};

MP_REGISTER_MODULE(MP_QSTR_digit_classifier, digit_classifier_module);
