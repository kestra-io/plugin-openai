package io.kestra.plugin.openai;

import io.kestra.core.runners.RunContext;
import io.kestra.core.runners.RunContextFactory;
import io.micronaut.test.extensions.junit5.annotation.MicronautTest;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.*;


@MicronautTest
@Disabled("Needs an OpenAI API Key to work")
public class CreateImageTest {
    @Inject
    private RunContextFactory runContextFactory;

    private String apiKey = "";


    @Test
    void runPromptUrl() throws Exception {
        RunContext runContext = runContextFactory.of();

        CreateImage task = CreateImage.builder()
            .apiKey(this.apiKey)
            .prompt("A funny cat in a black suit")
            .size(CreateImage.SIZE.SMALL)
            .build();

        CreateImage.Output runOutput = task.run(runContext);

        assertThat(runOutput.getImages().size(), is(1));
    }

    @Test
    void runPromptB64Json() throws Exception {
        RunContext runContext = runContextFactory.of();

        CreateImage task = CreateImage.builder()
            .apiKey(this.apiKey)
            .prompt("A funny cat in a black suit")
            .size(CreateImage.SIZE.SMALL)
            .download(true)
            .build();

        CreateImage.Output runOutput = task.run(runContext);

        assertThat(runOutput.getImages().size(), is(1));
    }
}
