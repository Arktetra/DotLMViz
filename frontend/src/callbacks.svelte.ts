import { getAct, getMLPOuts, getProbDensity, getTokens, runModel } from "./routes/fetch.svelte";
import { activeComponent, global_state, input } from "./state.svelte"

/**
 * A callback function that is to be called each time the input is changed.
 */
export const inputCallback = async (v: string) => {
    input.text = v;
    await getTokens(input.text);
    input.isChanged = true;
}

/**
 * A callback function that is to be called each time the token embedding
 * is clicked.
 */
export const embedCallback = async () => {
    if (input.isChanged === true) {
        await runModel(input.text);
    }
    await getAct("embed", null, null);
    activeComponent.name = "embed";
}

/**
 * A callback function that is to be called each time the position
 * embedding is clicked.
 */
export const posEmbedCallback = async () => {
    if (input.isChanged === true) {
        await runModel(input.text);
    }
    await getAct("pos_embed", null, null);
    activeComponent.name = "pos_embed";
}

/**
 * A callback function that is to be called each time the attention heads
 * are clicked.
 */
export const attnHeadCallback = async () => {
    if (input.isChanged === true) {
        await runModel(input.text);
    }
    await getAct("pattern", "attn", global_state.active_block);
    activeComponent.name = "attn";
}

/**
 * A callback function that is to be called each time the first layer of the
 * MLP is clicked.
 */
export const MLPPreCallback = async () => {
    if (input.isChanged === true) {
        await runModel(input.text);
    }
    await getMLPOuts("pre", "mlp", global_state.active_block, global_state.neuron);
    activeComponent.name = "mlp_pre";
}

/**
 * A callback function that is to be called each time the LN1 is clicked.
 */
export const LN1Callback = async () => {
    if (input.isChanged === true) {
        await runModel(input.text);
    }

    await getProbDensity("resid_pre", null, global_state.active_block);
    await getProbDensity("normalized", "ln1", global_state.active_block);

    activeComponent.name = "ln1";
}

/**
 * A callback function that is to be called each time the LN2 is clicked.
 */
export const LN2Callback = async () => {
    if (input.isChanged === true) {
        await runModel(input.text);
    }

    await getProbDensity("resid_mid", null, global_state.active_block);
    await getProbDensity("normalized", "ln2", global_state.active_block);

    activeComponent.name = "ln2";
}