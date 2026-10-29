package io.kestra.plugin.openai;

import java.net.URI;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.Locale;

import org.slf4j.Logger;

import com.openai.client.OpenAIClient;
import com.openai.models.files.FileCreateParams;
import com.openai.models.files.FilePurpose;

import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.annotations.PluginProperty;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.RunnableTask;
import io.kestra.core.runners.RunContext;
import io.kestra.core.utils.FileUtils;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.*;
import lombok.experimental.SuperBuilder;

@SuperBuilder
@ToString
@EqualsAndHashCode
@Getter
@NoArgsConstructor
@Schema(
    title = "Upload files to OpenAI storage",
    description = "Sends a file from Kestra internal storage to OpenAI for assistants, fine-tune, search, or other purposes, returning the file ID. Uses a temp copy locally before upload; unknown purposes fall back to `assistants`."
)
@Plugin(
    examples = {
        @Example(
            full = true,
            title = "Upload a file to OpenAI and get its ID for later use.",
            code = """
                id: openai_file_upload
                namespace: company.team

                inputs:
                  - id: dataFile
                    type: FILE

                tasks:
                  - id: upload_file
                    type: io.kestra.plugin.openai.UploadFile
                    apiKey: "{{ secret('OPENAI_API_KEY') }}"
                    from: "{{ inputs.dataFile }}"
                    purpose: "fine-tune"

                  - id: use_file_id
                    type: io.kestra.plugin.core.log.Log
                    message: "Uploaded file ID: {{ outputs.upload_file.fileId }}"
                """
        ),
        @Example(
            full = true,
            title = "Upload a user data file with description for user data purposes",
            code = """
                id: openai_search_file
                namespace: company.team

                inputs:
                  - id: documentFile
                    type: FILE

                tasks:
                  - id: upload
                    type: io.kestra.plugin.openai.UploadFile
                    apiKey: "{{ secret('OPENAI_API_KEY') }}"
                    from: "{{ inputs.documentFile }}"
                    purpose: "user_data"
                    description: "Customer knowledge base articles for retrieval"
                """
        )
    }
)
public class UploadFile extends AbstractTask implements RunnableTask<UploadFile.Output> {

    @Schema(
        title = "Source file URI",
        description = "Required `kestra://` or other storage URI pointing to the file to upload."
    )
    @NotNull
    @PluginProperty(internalStorageURI = true, group = "main")
    private Property<String> from;

    @Schema(
        title = "Upload purpose",
        description = "Accepted values include assistants, fine-tune, vision, user_data, evals, batch; defaults to assistants for unrecognized values."
    )
    @NotNull
    @PluginProperty(group = "main")
    private Property<String> purpose;

    @Override
    public UploadFile.Output run(RunContext runContext) throws Exception {
        Logger logger = runContext.logger();

        OpenAIClient client = this.openAIClient(runContext);

        String purpose = runContext.render(this.purpose).as(String.class).orElseThrow();
        String renderedFrom = runContext.render(this.from).as(String.class).orElseThrow();

        URI from = URI.create(renderedFrom);
        String extension = FileUtils.getExtension(from);

        var tempFile = runContext.workingDir().createTempFile(extension);

        try (var in = runContext.storage().getFile(from)) {
            Files.copy(in, tempFile, StandardCopyOption.REPLACE_EXISTING);
        }

        var fileObject = client.files().create(
            FileCreateParams.builder()
                .file(tempFile)
                .purpose(getFilePurpose(purpose))
                .build()
        );

        return Output.builder().fileId(fileObject.id()).build();
    }

    static FilePurpose getFilePurpose(final String purpose) {
        return switch (purpose.toLowerCase(Locale.ROOT)) {
            case "assistants" -> FilePurpose.ASSISTANTS;
            case "fine-tune" -> FilePurpose.FINE_TUNE;
            case "vision" -> FilePurpose.VISION;
            case "user_data" -> FilePurpose.USER_DATA;
            case "evals" -> FilePurpose.EVALS;
            case "batch" -> FilePurpose.BATCH;
            default -> FilePurpose.ASSISTANTS;
        };
    }

    @Builder
    @Getter
    public static class Output implements io.kestra.core.models.tasks.Output {
        @Schema(
            title = "Uploaded file ID"
        )
        private String fileId;
    }
}
