package io.kestra.plugin.openai;

import io.kestra.core.junit.annotations.KestraTest;
import org.assertj.core.util.Strings;

@KestraTest
public class AbstractOpenAITest {
    private static final String OPENAI_API_KEY = System.getenv("OPENAI_API_KEY");
    /* Create the following Prompt for testing:
    <system>
    Respond with the uppercased text that was provided by the user.
    Example:
    - User: qwer!
    - Assistant: QWER!
    </system>
     */
    private static final String PROMPT_ID_TO_UPPER = System.getenv("OPENAI_PROMPT_ID_TO_UPPER");
    /* Create the following Prompt for testing:
    <system>
    Respond with the uppercased text that was provided by the user,
    appending the user-provided suffix to the output (see below).
    Example:
    - User: qwer (provided suffix: ty)
    - Assistant: QWERty
    </system>
    <user>
    Provided suffix: {{suffix}}
    </user>
     */
    private static final String PROMPT_ID_TO_UPPER_SUFFIX = System.getenv("OPENAI_PROMPT_ID_TO_UPPER_SUFFIX");

    protected static boolean canNotBeEnabled() {
        return Strings.isNullOrEmpty(getApiKey());
    }

    protected static String getApiKey() {
        return OPENAI_API_KEY;
    }

    protected static boolean canNotTestPromptId() {
        return Strings.isNullOrEmpty(getPromptIdToUpper());
    }

    protected static String getPromptIdToUpper() {
        return PROMPT_ID_TO_UPPER;
    }

    protected static boolean canNotTestPromptIdVariables() {
        return Strings.isNullOrEmpty(getPromptIdToUpperSuffix());
    }

    protected static String getPromptIdToUpperSuffix() {
        return PROMPT_ID_TO_UPPER_SUFFIX;
    }
}
