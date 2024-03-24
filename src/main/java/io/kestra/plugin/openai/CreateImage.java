package io.kestra.plugin.openai;

import com.theokanning.openai.image.CreateImageRequest;
import com.theokanning.openai.image.ImageResult;
import com.theokanning.openai.service.OpenAiService;
import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.tasks.RunnableTask;
import io.kestra.core.runners.RunContext;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.*;
import lombok.experimental.SuperBuilder;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.UUID;

import jakarta.validation.constraints.NotNull;

import static io.kestra.core.utils.Rethrow.throwConsumer;

@SuperBuilder
@ToString
@EqualsAndHashCode
@Getter
@NoArgsConstructor
@Schema(
    title = "Given a prompt, create an image.",
    description = "For more information, refer to the [OpenAI Image Generation API docs](https://platform.openai.com/docs/api-reference/images/create)."
)
@Plugin(
    examples = {
        @Example(
            code = """
                prompt: A funny cat in a black suit
                apiKey: <your-api-key>
                download: true
                n: 5"""
        )
    }
)
public class CreateImage extends AbstractTask implements RunnableTask<CreateImage.Output> {
    @Schema(
        title = "Message to send to the API as prompt."
    )
    @NotNull
    @PluginProperty(dynamic = true)
    private String prompt;

    @Schema(
        title = "The number of images to generate. Must be between 1 and 10."
    )
    private Integer n;

    @Schema(
        title = "The size of the generated images."
    )
    @Builder.Default
    @PluginProperty
    private SIZE size = SIZE.LARGE;

    @Schema(
        title = "Whether to download the generated image",
        description = "If enable, the generated image will be downloaded inside Kestra's internal storage. Else, the URL of the generated image will be available as task output."
    )
    @Builder.Default
    @PluginProperty
    private boolean download = false;

    @Override
    public CreateImage.Output run(RunContext runContext) throws Exception {
        OpenAiService client = this.client(runContext);

        String user = runContext.render(this.user);
        String prompt = runContext.render(this.prompt);

        ImageResult imageResult = client.createImage(CreateImageRequest.builder()
            .prompt(prompt)
            .size(this.size.getSize())
            .n(this.n)
            .responseFormat(download ? "b64_json" : "url")
            .user(user)
            .build()
        );

        List<URI> files = new ArrayList<>();
        imageResult.getData().forEach(throwConsumer(image -> {
            if (download) {
                files.add(runContext.putTempFile(this.downloadB64Json(image.getB64Json())));
            } else {
                files.add(URI.create(image.getUrl()));
            }
        }));

        return Output.builder().images(files).build();
    }

    @Builder
    @Getter
    public static class Output implements io.kestra.core.models.tasks.Output {

        @Schema(
            title = "Generated images."
        )
        private List<URI> images;
    }

    private File downloadB64Json(String encodedImage) throws IOException {
        File image = File.createTempFile("openai-" + UUID.randomUUID(), ".png");
        byte[] decodedBytes = Base64.getDecoder().decode(encodedImage);
        FileUtils.writeByteArrayToFile(image, decodedBytes);

        return image;
    }

    protected enum SIZE {
        SMALL("256x256"),
        MEDIUM("512x512"),
        LARGE("1024x1024");

        private final String value;

        SIZE(String value) {
            this.value = value;
        }

        public String getSize() {
            return value;
        }

    }

}
