"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.trajectoryUnorderedMatch = void 0;
const utils_js_1 = require("../utils.js");
const utils_js_2 = require("./utils.js");
/**
 * Evaluate whether an input agent trajectory and called tools contains all the tools used in a reference trajectory.
 * This accounts for some differences in an LLM's reasoning process in a case-by-case basis.
 *
 * @param params - The parameters for trajectory unordered match evaluation
 * @param params.outputs - Actual trajectory the agent followed.
 *    May be a list of OpenAI messages, a list of LangChain messages, or a dictionary containing
 *    a "messages" key with one of the above.
 * @param params.reference_outputs - Ideal reference trajectory the agent should have followed.
 *    May be a list of OpenAI messages, a list of LangChain messages, or a dictionary containing
 *    a "messages" key with one of the above.
 * @returns EvaluatorResult containing a score of true if trajectory (including called tools) matches, false otherwise
 */
async function trajectoryUnorderedMatch(params) {
    const { outputs, referenceOutputs } = params;
    const outputsList = (0, utils_js_1._normalizeToOpenAIMessagesList)(outputs);
    const referenceOutputsList = (0, utils_js_1._normalizeToOpenAIMessagesList)(referenceOutputs);
    const getScore = async () => {
        if (outputsList == null || referenceOutputsList == null) {
            throw new Error("Trajectory unordered match requires both outputs and reference_outputs");
        }
        const unorderedMatch = (0, utils_js_2._isTrajectorySuperset)(outputsList, referenceOutputsList) &&
            (0, utils_js_2._isTrajectorySuperset)(referenceOutputsList, outputsList);
        return unorderedMatch;
    };
    return (0, utils_js_1._runEvaluator)("trajectory_unordered_match", getScore, "trajectory_unordered_match", params);
}
exports.trajectoryUnorderedMatch = trajectoryUnorderedMatch;
