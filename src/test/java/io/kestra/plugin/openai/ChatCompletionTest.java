package io.kestra.plugin.openai;

import com.theokanning.openai.completion.chat.ChatFunctionCall;
import com.theokanning.openai.completion.chat.ChatMessage;
import io.kestra.core.runners.RunContext;
import io.kestra.core.runners.RunContextFactory;
import io.kestra.core.junit.annotations.KestraTest;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;


@KestraTest
@Disabled("Needs an OpenAI API Key to work")
public class ChatCompletionTest {
    @Inject
    private RunContextFactory runContextFactory;

    private String apiKey = "";

    @Test
    void runMessages() throws Exception {
        RunContext runContext = runContextFactory.of();

        List<ChatMessage> messages = List.of(
            buildMessage("user","what is the capital of France?")
        );

        ChatCompletion task = ChatCompletion.builder()
            .apiKey(this.apiKey)
            .model("gpt-3.5-turbo-0613")
            .clientTimeout(30)
            .messages(messages)
            .build();

        ChatCompletion.Output runOutput = task.run(runContext);

        assertThat(runOutput.getChoices().get(0).getMessage().getContent(), containsString("Paris"));
        assertThat(runOutput.getModel(), containsString("gpt-3.5-turbo"));
        assertThat(runOutput.getUsage().getPromptTokens(), is(14L));
    }

    @Test
    void runPrompt() throws Exception {
        RunContext runContext = runContextFactory.of();

        ChatCompletion task = ChatCompletion.builder()
            .apiKey(this.apiKey)
            .model("gpt-3.5-turbo")
            .prompt("what is the capital of France?")
            .build();

        ChatCompletion.Output runOutput = task.run(runContext);

        assertThat(runOutput.getChoices().get(0).getMessage().getContent(), containsString("Paris"));
        assertThat(runOutput.getModel(), containsString("gpt-3.5-turbo"));
        assertThat(runOutput.getUsage().getPromptTokens(), is(14L));
    }

    @Test
    void runMessagesWithPrompt() throws Exception {
        RunContext runContext = runContextFactory.of();

        List<ChatMessage> messages = List.of(
            buildMessage("user","what is the capital of France?"),
            buildMessage("assistant","The capital of France is Paris.")
        );

        ChatCompletion task = ChatCompletion.builder()
            .apiKey(this.apiKey)
            .model("gpt-3.5-turbo-0613")
            .messages(messages)
            .prompt("and the capital of germany?")
            .build();

        ChatCompletion.Output runOutput = task.run(runContext);

        assertThat(runOutput.getChoices().get(0).getMessage().getContent(), containsString("Berlin"));
        assertThat(runOutput.getModel(), containsString("gpt-3.5-turbo"));
        assertThat(runOutput.getUsage().getPromptTokens(), is(35L));
    }

    @Test
    void runFunction() throws Exception {
        RunContext runContext = runContextFactory.of();

        List<ChatCompletion.PluginChatFunctionParameter> parameters = List.of(
            ChatCompletion.PluginChatFunctionParameter.builder()
                .name("location")
                .type("string")
                .description("The city and state/province, and country, e.g. San Francisco, CA, USA")
                .required(true)
                .build(),
            ChatCompletion.PluginChatFunctionParameter.builder()
                .name("unit")
                .type("string")
                .description("The temperature unit this city uses.")
                .required(true)
                .build()
        );

        List<ChatCompletion.PluginChatFunction> functions = List.of(
            ChatCompletion.PluginChatFunction.builder()
                .name("test")
                .description("finds the most relevant city and its temperature unit")
                .parameters(parameters)
                .build()

        );

        List<ChatMessage> messages = List.of(
            buildMessage("user","I was travelling along the Mediterranean coast, and I ended up in Lyon.")
        );

        ChatCompletion task = ChatCompletion.builder()
            .apiKey(this.apiKey)
            .model("gpt-3.5-turbo")
            .messages(messages)
            .functions(functions)
            .functionCall("auto")
            .build();

        ChatCompletion.Output runOutput = task.run(runContext);
        ChatFunctionCall functionCall = runOutput.getChoices().get(0).getMessage().getFunctionCall();

        assertThat(functionCall.getName(), containsString("test"));
        assertThat(functionCall.getArguments().get("location").toString(), containsString("Lyon"));
        assertThat(runOutput.getModel(), containsString("gpt-3.5-turbo"));
    }

    @Test
    void runFunctionWithEnumValues() throws Exception {
        RunContext runContext = runContextFactory.of();

        List<ChatCompletion.PluginChatFunctionParameter> parameters = List.of(
            ChatCompletion.PluginChatFunctionParameter.builder()
                .name("rating")
                .type("string")
                .description("A rating of what the customer thought of our restaurant based on the review they wrote.")
                .required(true)
                .enumValues(List.of("excellent", "acceptable", "terrible"))
                .build(),
            ChatCompletion.PluginChatFunctionParameter.builder()
                .name("food_eaten")
                .type("string")
                .description("A list of the food the customer ate, or 'Unknown' if they did not specify.")
                .required(true)
                .build(),
            ChatCompletion.PluginChatFunctionParameter.builder()
                .name("customer_name")
                .type("string")
                .description("The customer's name.")
                .required(true)
                .build()
        );

        List<ChatCompletion.PluginChatFunction> functions = List.of(
            ChatCompletion.PluginChatFunction.builder()
                .name("record_customer_rating")
                .description("Saves a customer's rating of our restaurant based on what they wrote in an online review.")
                .parameters(parameters)
                .build()
        );

        List<ChatMessage> messages = List.of(
            buildMessage("user","My name is John Smith. I ate at your restaurant last week and order the steak. It was the worst steak I've even eaten. I will not be returning!")
        );

        ChatCompletion task = ChatCompletion.builder()
            .apiKey(this.apiKey)
            .model("gpt-3.5-turbo")
            .messages(messages)
            .functions(functions)
            .functionCall("record_customer_rating")
            .build();

        ChatCompletion.Output runOutput = task.run(runContext);
        ChatFunctionCall functionCall = runOutput.getChoices().get(0).getMessage().getFunctionCall();

        assertThat(functionCall.getName(), containsString("record_customer_rating"));
        assertThat(functionCall.getArguments().get("rating").toString(), containsString("terrible"));
        assertThat(functionCall.getArguments()
            .get("food_eaten").toString()
            .toLowerCase(), containsString("steak"));
        assertThat(functionCall.getArguments().get("customer_name").toString(), containsString("John Smith"));
        assertThat(runOutput.getModel(), containsString("gpt-3.5-turbo"));
    }

    private ChatMessage buildMessage(String role, String content) {
        ChatMessage message = new ChatMessage();
        message.setRole(role);
        message.setContent(content);

        return message;
    }
}
