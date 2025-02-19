"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.trajectorySuperset = void 0;
const utils_js_1 = require("../utils.js");
const utils_js_2 = require("./utils.js");
/**
 * Evaluate whether an agent trajectory and called tools is a superset of a reference trajectory and called tools.
 * This means the agent called a superset of the tools specified in the reference trajectory.
 *
 * @param params - The parameters for trajectory superset evaluation
 * @param params.outputs - Actual trajectory the agent followed.
 *    May be a list of OpenAI messages, a list of LangChain messages, or a dictionary containing
 *    a "messages" key with one of the above.
 * @param params.reference_outputs - Ideal reference trajectory the agent should have followed.
 *    May be a list of OpenAI messages, a list of LangChain messages, or a dictionary containing
 *    a "messages" key with one of the above.
 * @returns EvaluatorResult containing a score of true if trajectory (including called tools) matches, false otherwise
 */
async function trajectorySuperset(params) {
    const { outputs, referenceOutputs } = params;
    const outputsList = (0, utils_js_1._normalizeToOpenAIMessagesList)(outputs);
    const referenceOutputsList = (0, utils_js_1._normalizeToOpenAIMessagesList)(referenceOutputs);
    const getScore = async () => {
        if (outputsList == null || referenceOutputsList == null) {
            throw new Error("Trajectory superset match requires both outputs and reference_outputs");
        }
        const isSuperset = (0, utils_js_2._isTrajectorySuperset)(outputsList, referenceOutputsList);
        return isSuperset;
    };
    return (0, utils_js_1._runEvaluator)("trajectory_superset", getScore, "trajectory_superset", params);
}
exports.trajectorySuperset = trajectorySuperset;
