"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.trajectoryStrictMatch = void 0;
const utils_js_1 = require("../utils.js");
function _scorer(params) {
    const { outputs, referenceOutputs, toolCallArgsExactMatch, messageContentExactMatch } = params;
    const normalizedOutputs = (0, utils_js_1._normalizeToOpenAIMessagesList)(outputs);
    const normalizedReferenceOutputs = (0, utils_js_1._normalizeToOpenAIMessagesList)(referenceOutputs);
    if (!normalizedOutputs || !normalizedReferenceOutputs) {
        throw new Error("Strict trajectory match requires both outputs and reference_outputs");
    }
    if (normalizedOutputs.length !== normalizedReferenceOutputs.length) {
        return false;
    }
    let exactMatch = true;
    for (let i = 0; i < normalizedOutputs.length; i++) {
        const output = normalizedOutputs[i];
        const referenceOutput = normalizedReferenceOutputs[i];
        if (output.role !== referenceOutput.role) {
            exactMatch = false;
            break;
        }
        const outputHasToolCalls = output.tool_calls != null;
        const referenceHasToolCalls = referenceOutput.tool_calls != null;
        if (outputHasToolCalls !== referenceHasToolCalls) {
            exactMatch = false;
            break;
        }
        if (outputHasToolCalls) {
            if (output.tool_calls.length !== referenceOutput.tool_calls.length) {
                exactMatch = false;
                break;
            }
            for (let j = 0; j < output.tool_calls.length; j++) {
                if (output.tool_calls[j].function.name !==
                    referenceOutput.tool_calls[j].function.name) {
                    exactMatch = false;
                    break;
                }
                if (toolCallArgsExactMatch &&
                    output.tool_calls[j].args !==
                        referenceOutput.tool_calls[j].args) {
                    exactMatch = false;
                    break;
                }
            }
        }
        if (messageContentExactMatch &&
            output.content !== referenceOutput.content) {
            exactMatch = false;
            break;
        }
    }
    return exactMatch;
}
async function trajectoryStrictMatch(params) {
    /**
     * Evaluate whether an input agent trajectory and called tools strictly matches a reference trajectory.
     * This means that at each step, the agent called the same tools in the same order as specified in the reference trajectory.
     *
     * @param outputs - Actual trajectory the agent followed. May be a list of OpenAI messages,
     *                 a list of LangChain messages, or a dictionary containing a "messages" key with one of the above.
     * @param referenceOutputs - Ideal reference trajectory the agent should have followed. May be a list of OpenAI messages,
     *                          a list of LangChain messages, or a dictionary containing a "messages" key with one of the above.
     * @param toolCallArgsExactMatch - Whether to require exact matches for tool call arguments
     * @param messageContentExactMatch - Whether to require exact matches for message content
     * @returns EvaluatorResult containing a score of true if trajectory (including called tools) matches, false otherwise
     */
    function _wrapper() {
        return _scorer(params);
    }
    return (0, utils_js_1._runEvaluator)("trajectory_strict_match", _wrapper, "trajectory_strict_match", params);
}
exports.trajectoryStrictMatch = trajectoryStrictMatch;
